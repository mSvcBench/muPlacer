# GMA Strategy Connector 
The `Strategy_Connector` component is a Python code that implements the placement strategy. The `Strategy_Connector` should include a function called `Compute_Placement` whose interface is 
```python
 def Compute_Placement(GMA_params, action='offloading'):
    ...
    return result_dict 
```
The `GMA_params` is an input dictionary passed by GMA to the `Compute_Placement` function that contains various metrics and configurations for microservices deployed in a cloud-edge infrastructure. The `action` is a string that can be either 'offloading' or 'unoffloading'.
The `result_dict` is a dictionary that contains the result of the placement strategy. 

## GMA_params dictionary

The `GMA_params` contains various metrics and configurations for microservices deployed in a cloud-edge infrastructure. 

Metrics can be scalar, vector and matrix values. For vector, the element *i* stores the metric for the microservice *i*. For matrix, the element *i,j* stores the metric for the couple of microservice *i,j*. 

The snippets report initialization values that are updated by GMA during the execution by using Istio and Kubernetes metrics.


### Number of Microservices

The number of microservices *M* is set using the `n-services` key within the `service-metrics` dictionary. This value includes the user, who is virtually represented as the last microservice.


```python
GMA_params['n-services'] 
```

### Call Frequency Matrix

The call frequency matrix is a square MxM numpy matrix that records the frequency of calls between microservices. The element (*i,j*) represent the average number of times microservice *i* calls microservice *j* per request received by *i*. It is initialized as follows:

```python
GMA_params['fm'] = {
    'info': 'Call frequency matrix',
    'value': np.zeros((M, M), dtype=float),
    'last-update': 0  # last update time
}
```

### Response Length Vector

The response length vector records the size of responses in bytes for each microservice:

```python
GMA_params['response-length'] = {
    'info': 'Response size vector in bytes',
    'value': np.zeros(M, dtype=float),
    'last-update': 0  # last update time
}
```

### Horizontal Pod Autoscaler (HPA) Metrics Vector

HPA metrics include replica counts and CPU thresholds for both cloud and edge areas:

```python
GMA_params['hpa'] = {
    'cloud-area': {
        'info': 'HPA vectors for cloud area',
        'current-replicas': np.zeros(M, dtype=int),
        'desired-replicas': np.zeros(M, dtype=int),
        'old-current-replicas': np.zeros(M, dtype=int),
        'min-replicas': np.zeros(M, dtype=int),
        'max-replicas': np.zeros(M, dtype=int),
        'cpu-threshold': np.ones(M, dtype=int) * 0.6,
        'last-update': 0  # last update time
    },
    'edge-area': {
        'info': 'Replicas vector for edge area',
        'current-replicas': np.zeros(M, dtype=int),
        'desired-replicas': np.zeros(M, dtype=int),
        'old-current-replicas': np.zeros(M, dtype=int),
        'min-replicas': np.zeros(M, dtype=int),
        'max-replicas': np.zeros(M, dtype=int),
        'cpu-threshold': np.ones(M, dtype=int) * 0.6,
        'last-update': 0  # last update time
    }
}
```

### Cumulative CPU and Memory Utilization Vector

CPU and memory utilization metrics are recorded for both cloud and edge areas. CPU utilization is measured in seconds per second, while memory utilization is measured in bytes. Their values include all running Pod of a microservice:

```python
GMA_params['ucpu'] = {
    'cloud-area': {
        'info': 'Actual CPU utilization vector in seconds per second for cloud area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    },
    'edge-area': {
        'info': 'Actual CPU utilization vector in seconds per second for edge area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    }
}

GMA_params['umem'] = {
    'cloud-area': {
        'info': 'Actual memory utilization vector in bytes for cloud area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    },
    'edge-area': {
        'info': 'Actual memory utilization vector in bytes for edge area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    }
}
```

### CPU and Memory Request Vector

Requested CPU and memory per pod are recorded for both cloud and edge areas:

```python
GMA_params['qcpu'] = {
    'cloud-area': {
        'info': 'Requested CPU per pod in seconds per second for cloud area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    },
    'edge-area': {
        'info': 'Requested CPU per pod in seconds per second for edge area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    }
}

GMA_params['qmem'] = {
    'cloud-area': {
        'info': 'Requested memory per pod in bytes for cloud area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    },
    'edge-area': {
        'info': 'Requested memory per pod in bytes for edge area',
        'value': np.zeros(M, dtype=float),
        'last-update': 0  # last update time
    }
}
```

### Request Rate Vector

The request rate vector records the request rate in requests per second for each microservice. The user request rate is included in the last element of the vector and is measured through the Istio Ingress Gateway of the edge data center:

```python
GMA_params['service-lambda'] = {
    'info': 'Request rate vector in req/s',
    'value': np.zeros(M, dtype=float),
    'last-update': 0  # last update time
}
```

### Edge User Delay Metrics

Edge user delay metrics include average delay, delay quantile, and target delay. Delays are measured at the Istio Ingress Gateway of the edge data center:

```python
GMA_params['edge-user-delay'] = {
    'value': 0.0,  # last update time
    'info': 'Average edge user delay in ms',
    'last-update': 0  # last update time
}

GMA_params['edge-user-delay-quantile'] = {
    'value': 0.0,
    'info': 'Edge user delay quantile in ms',
    'last-update': 0  # last update time
}

GMA_params['edge-user-target-delay'] = {
    'value': 0.0,  # last update time
    'info': 'Average edge user target delay in ms',
    'last-update': 0  # last update time
}
```

### Network Metrics

Network metrics include network round-trip time and bitrate between edge and cloud areas. The RTT multiplier is applied to network RTT to obtain gRPC/HTTP-level round-trip time:

```python
GMA_params['network'] = {
    'edge-cloud-rtt-ms': {
        'value': time_to_ms_converter(gma_config['spec']['network']['edge-cloud-rtt-ms']),
        'info': 'Round trip time from edge area to cloud area in ms',
        'last-update': 0  # last update time
    },
        'edge-cloud-rtt-multiplier': {
        'value': gma_config['spec']['network']['edge-cloud-rtt-multiplier'],
        'info': 'The RTT multiplier is applied to network RTT to obtain gRPC/HTTP-level round-trip time. Depends on the application. Configure with offline measurements',
        'last-update': 0  # last update time
    },
    'cloud-edge-bps': {
        'value': bitrate_to_bps_converter(gma_config['spec']['network']['cloud-edge-bps']),
        'info': 'Network capacity in bits per second from cloud area to edge area in bps',
        'last-update': 0  # last update time
    },
    'edge-cloud-bps': {
        'value': bitrate_to_bps_converter(gma_config['spec']['network']['edge-cloud-bps']),
        'info': 'Network capacity in bits per second from edge area to cloud area in bps',
        'last-update': 0  # last update time
    }
}
```

### Cost Metrics

Cost metrics include the cost of CPU, memory, and network for both edge and cloud areas:

```python
GMA_params['cost'] = {
    'edge-area': {
        'cpu': {
            'value': gma_config['spec']['edge-area']['cost']['cpu'],
            'info': 'Cost of CPU in the edge area per hour'
        },
        'memory': {
            'value': gma_config['spec']['edge-area']['cost']['memory'],
            'info': 'Cost of memory in the edge area per GB'
        },
        'network': {
            'value': gma_config['spec']['edge-area']['cost']['memory'],
            'info': 'Cost of external network traffic for the edge area per GB'
        }
    },
    'cloud-area': {
        'cpu': {
            'value': gma_config['spec']['cloud-area']['cost']['cpu'],
            'info': 'Cost of CPU in the cloud area per hour'
        },
        'memory': {
            'value': gma_config['spec']['cloud-area']['cost']['memory'],
            'info': 'Cost of memory in the cloud area per GB'
        },
        'network': {
            'value': gma_config['spec']['cloud-area']['cost']['memory'],
            'info': 'Cost of external network traffic for the cloud area per GB'
        }
    }
}
```

### Multi-Edge Resource Scaling

The multi-edge resource scaling factor is a vector that scales resources between cloud and edge areas. It should be equal to the ratio between request rate received by the edge data center associated to the GMA and the cumulative request rate from all edge data centers. It is initialized as follows:

```python
GMA_params['me-resource-scaling'] = {
    'info': 'Cloud-to-edge multi-edge resource scaling factor',
    'value': np.ones(M, dtype=float) * gma_config['spec']['edge-area']['default-resource-scaling'],
    'last-update': 0  # last update time
}
```

### Optimizer information

GMA_params includes the optimizer information of the GMA configuration `gma_config['spec']['optimizer']`:    
    
```python
GMA_params['optimizer'] = gma_config['spec']['optimizer'].copy()
```


## Return result_dict

The `result_dict` is a `list` of dictionaries that contains the result of the placement strategy for cloud and edge areas. The following code snippet describes the structure of the output data that are returned by the `Compute_Placement` component to GMA.
```python

    result_cloud = dict()
    result_cloud['to-apply'] = list() # list of microservice ids to apply in the cloud area
    result_cloud['to-delete'] = list() # list of microservice ids to delete in the cloud area
    result_cloud['placement'] = list() # list of microservice ids in the cloud area
    result_cloud['info'] = f"Result for offloading - cloud microservice ids: {result_cloud['placement']}"

    result_edge = dict()
    result_edge['to-apply'] = list() # list of microservice ids to apply in the edge area
    result_edge['to-delete'] = list() # list of microservice ids to delete in the edge area
    result_edge['placement'] = list() # list of microservice ids in the edge area
    result_edge['info'] = f"Result for offloading - edge microservice ids: {result_edge['placement']}"
    
    result_dict=list()
    result_dict.append(result_cloud)  
    result_dict.append(result_edge)

```