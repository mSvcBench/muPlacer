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

The install istio-ingress on the edge cluster as it follows from the cloud master: 
```bash
    helm install istio-ingressgateway istio/gateway -n istio-ingress-edge1
```

To access istio-ingress is necessary to retrieve the NodePort and (possibly) LoadBalancer IP of the istio-ingressgateway service. To do this, run the following command from the cloud master:
```bash
    k get svc -n istio-ingress-edge1 --kubeconfig .kube/edge-config
``` 
