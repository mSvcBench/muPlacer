# GMA with LIQO multi-cluster
[Liqo](https://liqo.io) is a multi-cluster management platform that allows the dynamic sharing of services across Kubernetes clusters. It enables the creation of a single, virtual cluster to provide a unified view of the resources available in the connected clusters. 

Liqo is used in the GMA example to create a multi-cluster environment with a cloud cluster and an edge cluster. The cloud cluster runs the entire set of microservices, while the edge cluster dynamically runs a subset of microservices chosen by GMA to ensure that the average user delay remains within a higher value, defined as the *offload threshold*, and a lower value, defined as the *unoffload threshold*.

## Prepare the cloud and edge clusters
To prepare the cloud and edge clusters, follow the instructions in the [Liqo documentation](https://doc.liqo.io/user-guide/getting-started/). Shortly, you need to install Liqo on both clusters and establish a peering relationship between them.

In what follows we report an exaple of how to install Liqo on a cloud and edge1 cluster which uses Calico network plugin with VXLAN tunneling and exposes external service through NodePort, i.e. no LoadBalabcer is available.

> **Note:** Liqo uses port 6443 for the peering communication. Make sure these ports are open between the two clusters.
>
> 
### Install Liqo
With calico CNI it is necessary to disable BGP on liqo interfaces on clusters. To do this, run the following commands on both clusters: 
```bash
kubectl set env daemonset/calico-node -n kube-system IP_AUTODETECTION_METHOD=skip-interface=liqo.*
```

To **install Liqo** (v1.0.0-rc.3) on the cloud cluster, run the following commands from the master node:
```bash
curl --fail -LS "https://github.com/liqotech/liqo/releases/download/v1.0.0-rc.3/liqoctl-linux-amd64.tar.gz" | tar -xz
sudo install -o root -g root -m 0755 liqoctl /usr/local/bin/liqoctl
liqoctl install kubeadm --cluster-labels topology.kubernetes.io/zone=cloud --cluster-id cloud
```
And on the edge cluster, from the edge master node:
```bash
curl --fail -LS "https://github.com/liqotech/liqo/releases/download/v1.0.0-rc.3/liqoctl-linux-amd64.tar.gz" | tar -xz
sudo install -o root -g root -m 0755 liqoctl /usr/local/bin/liqoctl
liqoctl install kubeadm --cluster-labels topology.kubernetes.io/zone=edge1 --cluster-id edge1
```

### Cluster peering
To peer the two clusters, run the following command on the cloud cluster, where `.kube/edge-config` is the kubeconfig file of the edge cluster:
```bash
liqoctl peer --remote-kubeconfig .kube/edge-config --server-service-type NodePort
```

Revise the quota of resources that can be used on edge cluster, e.g., with:
```bash
liqoctl create resourceslice edge1 --remote-cluster-id edge1 --cpu 20 --memory 20Gi
```

### Node topology labeling
To support locality load balancing, it is necessary to label each node of the cloud and edge cluster with the `topology.kubernetes.io/zone` label. To do this, run the following commands on the cloud and edge clusters, respectively:
```bash
kubectl label nodes <cloud-node-name> topology.kubernetes.io/zone=cloud
```
```bash
kubectl label nodes <edge-node-name> topology.kubernetes.io/zone=edge1
```

### Install Istio

#### Install Istio control plane in the cloud cluster

First, we need to prepare Istio offloading in the cloud cluster only as it follows from the cloud master:
```bash
kubectl create namespace istio-system
liqoctl offload namespace istio-system  --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Local
```

Then install Istio control plane with the following commands on the cloud cluster:

```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update
kubectl create namespace istio-system
helm install istio-base istio/base -n istio-system
helm install istiod istio/istiod -n istio-system --wait
```

#### Install Istio-ingress on edge cluster
To install istio-ingress in a specific edge cluster we creade a dedicated namespace per edge cluster, offload them with Liqo, and install an istio ingress helm chart in the different namespace. In our case we have a single edge cluster denoted as `edge1`. 

First, we prepare istio-ingress offloading in the edge cluster as it follows from the cloud master:
```bash
kubectl create namespace istio-ingress-edge1
liqoctl offload namespace istio-ingress-edge1  --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Remote
```

Then install istio-ingress on the edge cluster as it follows from the cloud master: 
```bash
helm install istio-ingressgateway istio/gateway -n istio-ingress-edge1
```

To access istio-ingress is necessary to retrieve the NodePort and (possibly) LoadBalancer IP of the istio-ingressgateway service. To do this, run the following command from the cloud master:
```bash
kubectl get svc -n istio-ingress-edge1 --kubeconfig .kube/edge-config
``` 
### Install Prometheus on cloud cluster
Install Prometheus and monitoring tools on cloud cluster as it follows from the cloud master: 
```bash
cd examples/liqo/mubench-app/prometheus
sh monitoring.sh
cd ../../../..
```
The script install monitoring tools in the `monitoring`namespace and expose the following NodePorts:
- 30000 for Prometheus
- 30001 for Grafana (admin:prom-operator)
- 30002 for Jaeger
- 30003 for Kiali

### Install iperf3 server and netprober
Iperf3 and netprober are used for network testing by GMA. Install as it follows from the cloud master:
```bash
kubectl create namespace iperf-edge1
liqoctl offload namespace iperf-edge1  --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Remote
kubectl apply -f 'examples/liqo/iperf3/iperf3.yaml'

kubectl create namespace gma-netprober
kubectl apply -f 'netprober/netprober.yaml'
```

## Install sample application
We use a [µBench](https://github.com/mSvcBench/muBench) sample application made of 10 microservices. All next commands run from the cloud master.

### Application namespace offloading
The application run in the `fluidosmesh`namespace that need to be offloaded as follow:
```bash
liqoctl offload namespace fluidosmesh  --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy LocalAndRemote
```

### Application deployment
Deploy whole application with HPA in the cloud cluster with:
```bash
kubectl apply -f '/home/ubuntu/muPlacer/examples/liqo/mubench-app/affinity-yamls/no-region-specified/cloud/no-subzone-specified'
kubectl apply -f '/home/ubuntu/muPlacer/examples/liqo/mubench-app/hpa/cloud'
```

Allow istio-ingress access to microservice s0 and locality load balancing with:
```bash
kubectl apply -f k apply -f '/home/ubuntu/muPlacer/examples/liqo/mubench-app/dest-rule-yamls-least-request'
```

### GMA deployment 
#### Revise GMA configuration
GMA run as a Python process from which has kubectl and kubernetes contexts of cloud and edge1 clusters. Carefully revise the GMA configuration file `gma-config.yaml` where with your parameters. Critical value to revise are: 
 - ip address of a node in the cloud cluster providing NodePort access **192.168.100.142**, chage this value accordingly in `prometheus-url`and `netprober-url` fields.
 - kubernetes context of the cloud cluster that (with liqo) should be used also to control the edge cluster: kubernetes-admin@cluster.local
 - regex to match the pod cidr of the cloud area (must be different from any other area) : ^10.234.*
 - regex to match the pod cidr of the edge area (must be different from any other area) : ^10.236.*, **

#### Deploy GMA
It is necessary to create the Python environment with:
```bash
git clone https://github.com/mSvcBench/muPlacer.git
cd muPlacer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, run GMA with:
```bash
python3 GMA.py --config examples/liqo/GMAConfig.yaml --loglevel INFO
```

The GMA will start to monitor the application and the network, and it will offload the microservices to the edge cluster when the average user delay is above the offload threshold and will bring them back when the average user delay is below the unoffload threshold. Related states and actions are logged in the console.

#### Load testing
To test the application, we should a stream of send requests to the istio-ingressgateway service in the edge cluster. We used [Jmeter](https://jmeter.apache.org) tool with the configuration file `examples/jmeter/GMATest.jmx`, which can be used by any user host with access to the IP address and NodePort of the Istio ingress gateway of the edge cluster. The related command to run from the host is:
```bash
jmeter -Jserver=<edge-node-ip> -Jport=<istio-ingress-node-port> -Jthroughput=10 -n -t examples/jmeter/GMATest.jmx
```
The throughput parameter is the number of requests per second that the Jmeter will send to the edge cluster.

### Monitoring
To monitor the application behaviour, we use a Grafana dashboard that can be accessed at the address `http://<cloud-node-ip>:30001` with the credentials `admin:prom-operator`. The dashboard is available at `examples/liqo/grafana/edge-computing-liqo.json`. The dashboard has some variables that need to be set to the correct values. The variables are:
- istio_ingress_namespace : istio-ingress-edge1
- app_namespace: fluidosmesh
