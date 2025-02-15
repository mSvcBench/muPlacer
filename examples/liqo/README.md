# GMA with LIQO Multi-Cluster

[Liqo](https://liqo.io) is a multi-cluster management platform that allows the dynamic sharing of services across Kubernetes clusters. It enables the creation of a single, virtual cluster to provide a unified view of the resources available in the connected clusters.

Liqo is used in the GMA example to create a multi-cluster environment with a cloud cluster and an edge cluster. The cloud cluster runs the entire set of microservices, while the edge cluster dynamically runs a subset of microservices chosen by GMA to ensure that the average user delay remains within a higher value, defined as the *offload threshold*, and a lower value, defined as the *unoffload threshold*.

## Prepare the Cloud and Edge Clusters

To prepare the cloud and edge clusters, follow the instructions in the [Liqo documentation](https://doc.liqo.io/user-guide/getting-started/). In summary, you need to install Liqo on both clusters and establish a peering relationship between them.

Below is an example of how to install Liqo on a cloud and edge cluster using the Calico network plugin with VXLAN tunneling and exposing external services through NodePort (i.e., no LoadBalancer is available).

> **Note:** Liqo uses port 6443 for peering establishment and Kubernetes NodePorts 30000-32767 for gateway traffic. Ensure these ports are open between the two clusters.

> **Note:** GMA requires the use of different POD CIDR per clusters. Please, configure the network plugin accordingly.

### Install Liqo

With the Calico CNI, it is necessary to disable BGP on Liqo interfaces on the clusters. To do this, run the following command on both clusters:

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

### Cluster Peering

To peer the two clusters, run the following command on the cloud cluster, where `~/.kube/edge-config` is the kubeconfig file of the edge cluster:

```bash
liqoctl peer --remote-kubeconfig ~/.kube/edge-config --server-service-type NodePort
```

Revise the quota of resources that can be used on the edge cluster, for example:

```bash
liqoctl create resourceslice edge1 --remote-cluster-id edge1 --cpu 20 --memory 20Gi
```

### Node Topology Labeling

To support Istio locality load balancing, it is necessary to label each node of the cloud and edge clusters with the `topology.kubernetes.io/zone` label. To do this, run the following commands on the cloud and edge clusters, respectively:

```bash
kubectl label nodes <cloud-node-name> topology.kubernetes.io/zone=cloud
```

```bash
kubectl label nodes <edge-node-name> topology.kubernetes.io/zone=edge1
```

### Install Istio

#### Install Istio Control Plane in the Cloud Cluster

First, prepare cloud-only Istio control plane offloading as follows from the cloud master node:

```bash
kubectl create namespace istio-system
liqoctl offload namespace istio-system --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Local
```

Then install the Istio control plane. We use Helm with the following commands from the cloud master node:

```bash
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update
helm install istio-base istio/base -n istio-system
helm install istiod istio/istiod -n istio-system --set global.proxy.tracer="zipkin" --wait
```

#### Install Istio-Ingress on the Edge Cluster

To install Istio-Ingress in a specific edge cluster, create a dedicated namespace per edge cluster, offload them with Liqo, and install an Istio Ingress Helm chart in the different namespaces. In our case, we have a single edge cluster denoted as `edge1`.

First, prepare Istio-Ingress offloading in the edge cluster as follows from the cloud master:

```bash
kubectl create namespace istio-ingress-edge1
liqoctl offload namespace istio-ingress-edge1 --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy LocalAndRemote --selector 'topology.kubernetes.io/zone=edge1'
```

Then install Istio-Ingress on the edge cluster as follows from the cloud master:

```bash
helm install istio-ingressgateway istio/gateway -n istio-ingress-edge1
```

To access Istio-Ingress, retrieve the NodePort and (possibly) LoadBalancer IP of the `istio-ingressgateway` service. To do this, run the following command from the cloud master:

```bash
kubectl get svc -n istio-ingress-edge1 --kubeconfig ~/.kube/edge-config
```

### Install Prometheus and Telemetry tools on the Cloud Cluster

Install Prometheus and monitoring tools on the cloud cluster as follows from the cloud master:

```bash
cd examples/liqo/telemetry
sh monitoring.sh
cd ../../..
```

The script installs monitoring tools in the `monitoring` namespace and exposes the following NodePorts:
- 30000 for Prometheus
- 30001 for Grafana (admin:prom-operator)
- 30002 for Jaeger
- 30003 for Kiali

### Install iperf3 Server and Netprober

Iperf3, on edge cluster, and Netprober, on cloud cluster, are used for network probing by GMA. Install them as follows from the cloud master:

```bash
kubectl create namespace iperf-edge1
liqoctl offload namespace iperf-edge1 --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Remote
kubectl apply -f 'examples/liqo/iperf3/iperf3.yaml'

kubectl create namespace gma-netprober
liqoctl offload namespace gma-netprober --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy Local
kubectl apply -f 'netprober/netprober.yaml'
```

## Install Sample Application

We use the following sample applications:
- a [µBench](https://github.com/mSvcBench/muBench) sample application made of [10 microservices](mubench-app/servicegraph.png). 
- the onlinebutique [Google microservices demo](https://github.com/GoogleCloudPlatform/microservices-demo).

All subsequent commands run from the cloud master.

### Application Namespace Offloading and Istio injection

The application runs in the `fluidosmesh` namespace, which needs to be offloaded as follows:

```bash
liqoctl offload namespace fluidosmesh --namespace-mapping-strategy EnforceSameName --pod-offloading-strategy LocalAndRemote
```

Then, inject Istio sidecars in the `fluidosmesh` namespace:

```bash
kubectl label namespace fluidosmesh istio-injection=enabled
```

### µBench Application Deployment

Deploy the entire µBench application with HPAs in the cloud cluster with:

```bash
kubectl apply -n fluidosmesh -f 'examples/liqo/mubench-app/affinity-yamls/no-region-specified/cloud/no-subzone-specified'
kubectl apply -n fluidosmesh -f 'examples/liqo/mubench-app/hpa/cloud'
kubectl apply -n fluidosmesh -f 'examples/liqo/mubench-app/services'
```

Allow Istio-Ingress access to microservice `s0` and locality load balancing for any microservice with:

```bash
kubectl apply -n fluidosmesh -f '/home/ubuntu/muPlacer/examples/liqo/mubench-app/istio-resources'
```

#### GMA Deployment

##### Download GMA

Download GMA (e.g., on cloud master) and create the Python environment with:

```bash
git clone https://github.com/mSvcBench/muPlacer.git
cd muPlacer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##### Revise GMA Configuration
GMA runs as a Python process with kubectl and Kubernetes contexts of the cloud and edge clusters. Carefully revise the GMA configuration file [GMAConfig.yaml](examples/liqo/mubench-app/GMAConfig.yaml) in `examples/liqo/mubench-app`with your parameters. Critical values to revise are:
- The URL of the prometheus server (`prometheus-url`) that can be contacted by GMA, e.g., 192.168.100.142:30000. Change this value accordingly in the  and `netprober-url` fields.
- The URL of the netprober server (`netprober-url`) that can be contacted by GMA, e.g., http://192.168.100.142:30123
- The IP address of the <u>Pod</u> that run the iperf3 server in the edge cluster to be inserted in `server_ip` of `netprober-url`, e.g., 10.236.149.25. IP address of the Pod is necessary to support RTT estimation via ICMP. 
- The Kubernetes context of the <u>cloud cluster</u>, e.g. `kubernetes-admin@cluster.local`. Copy this value both in the `cloud-area.context` and `edge-area.context` fields.
- The regex to match the Pod CIDR of the cloud area (must be different from any other area): `^10.234.*`.
- The regex to match the Pod CIDR of the edge area (must be different from any other area): `^10.236.*`.

##### Run GMA
Run GMA with:

```bash
python3 GMA.py --config examples/liqo/mubench-app/GMAConfig.yaml --loglevel INFO
```

GMA will start monitoring the application and the network, and it will offload the microservices to the edge cluster when the average user delay is above the offload threshold. It will bring them back when the average user delay is below the unoffload threshold. Related states and actions are logged in the console.

#### Load Testing

To test the µBench application, send a stream of requests to the Istio-Ingressgateway service in the edge cluster. 
##### JMeter
We used the [JMeter](https://jmeter.apache.org) tool with the configuration file `examples/liqo/mubench-app/jmeter/GMATest.jmx`, which can be used by any user host with access to the IP address and NodePort of the Istio Ingress gateway of the edge cluster. The related command to run from the host is:

```bash
jmeter -Jserver=<edge-node-ip> -Jport=<istio-ingress-node-port> -Jthroughput=10 -n -t examples/jmeter/GMATest.jmx
```

The `throughput` parameter is the number of requests per second that JMeter will send to the edge cluster.

##### Locust
Alternatively, we used the [Locust](https://locust.io) tool with the configuration file `examples/liqo/mubench-app/locust/locustfile.py`, which can be used by any user host with access to the IP address and NodePort of the Istio Ingress gateway of the edge cluster. The related command to run from the host is:

```bash
locust -f examples/liqo/mubench-app/locust/locustfile.py --host=http://<edge-node-ip>:<istio-ingress-node-port> --headless -u <n_users> -r <rate_per_user>
```
The `n_users` parameter is the number of users that Locust will simulate, and the `rate_per_user` parameter is the rate of requests per second that each user will send to the edge cluster. 

### Online Boutique Application Deployment
Onlinebutique is a Google microservices demo that we use to test the GMA with LIQO. The application is composed of 10 microservices and is deployed in the `fluidosmesh` namespace.

The original application uses a single Redis database. We modified the application to use a Redis database per cluster to avoid the need for a shared database across clusters. Specifically, we use a 'master' Redis database in the cloud cluster and a 'replica' Redis database in the edge cluster. In so doing, we avoid the need to reach cloud cluster when the application is deployed at the edge

To deploy the couple of Redis databases, run the following commands from the cloud master:

```bash
kubectl apply -n fluidosmesh -f 'examples/liqo/onlinebutique/redis'
```

Deploy the entire onlinebutique application with HPAs in the cloud cluster with:

```bash
kubectl apply -n fluidosmesh -f 'examples/liqo/onlinebutique/affinity-yamls/no-region-specified/cloud/no-subzone-specified'
kubectl apply -n fluidosmesh -f 'examples/liqo/onlinebutique/hpa/cloud'
kubectl apply -n fluidosmesh -f 'examples/liqo/onlinebutique/services'
```

Allow Istio-Ingress access to microservice `frontend` and locality load balancing for any microservice with:

```bash
kubectl apply -n fluidosmesh -f 'examples/liqo/onlinebutique/istio_resources'
```

#### GMA Run
Download and revise the GMA configuration file as described in the previous section. Run GMA with:

```bash
python3 GMA.py --config examples/liqo/onlinebutique/GMAConfig.yaml --loglevel INFO
```



#### Load Testing

To test the µBench application, send a stream of requests to the Istio-Ingressgateway service in the edge cluster. 

##### Locust
We used the [Locust](https://locust.io) tool with the configuration file `examples/liqo/onlinebutique/locust/locustfile.py`, which can be used by any user host with access to the IP address and NodePort of the Istio Ingress gateway of the edge cluster. The related command to run from the host is:

```bash
locust -f examples/liqo/mubench-app/locust/locustfile.py --host=http://<edge-node-ip>:<istio-ingress-node-port> --headless -u <n_users> -r <rate_per_user>
```
The `n_users` parameter is the number of users that Locust will simulate, and the `rate_per_user` parameter is the rate of requests per second that each user will send to the edge cluster.


## Monitoring

To monitor the application's behavior, use a Grafana dashboard that can be accessed at `http://<cloud-node-ip>:30001` with the credentials `admin:prom-operator`. The dashboard is available at `examples/liqo/grafana/edge-computing-liqo.json`. The dashboard has some variables that need to be set to the correct values. The variables are:
- `istio_ingress_namespace`: `istio-ingress-edge1`
- `app_namespace`: `fluidosmesh`
```
