import numpy as np

def Connector(GMA_params):
    # Take params from GMA and return EPAMP params
    M = GMA_params['n-services']
    S_edge_b = np.minimum(GMA_params['replicas']['edge-area']['value'],1)
    S_edge_b[M-1]=1 # Last service is always on the edge since it represent the istio-ingress/user
    Rcpu = np.zeros(2*M)
    Rcpu[0:M] = GMA_params['acpu']['cloud-area']['value']
    Rcpu[M:2*M] = GMA_params['acpu']['edge-area']['value']
    Rmem = np.zeros(2*M)
    Rmem[0:M] = GMA_params['amem']['cloud-area']['value']
    Rmem[M:2*M] = GMA_params['amem']['edge-area']['value']
    Fcm = GMA_params['fcm']['value']
    lambda_val = GMA_params['lambda']['value'][-1]
    Rs = GMA_params['rs']['value']
    if 'di' in GMA_params:
        Di = GMA_params['di']['value']
    else:
        Di = np.zeros(2*M)
    delay_decrease_target = GMA_params['delay']['value'][-1] - GMA_params['target-delay']['value'][-1]
    delay_increase_target = GMA_params['target-delay']['value'][-1]-GMA_params['delay']['value'][-1]
    Ne = GMA_params['network-capacity']['cloud-area']['value']
    RTT = GMA_params['rtt']['cloud-area']['value']/1000
    Cost_cpu_edge = GMA_params['cost']['edge-area']['cpu']['value']
    Cost_mem_edge = GMA_params['cost']['edge-area']['memory']['value']

    params = {
        'S_edge_b': S_edge_b,
        'Rcpu': Rcpu,
        'Rmem': Rmem,
        'Fcm': Fcm,
        'M': M,
        'lambd': lambda_val,
        'Rs': Rs,
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'delay_increase_target': delay_increase_target,
        'RTT': RTT,
        'Ne': Ne,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': None,
        'dependency_paths_b': None,
        'u_limit': 2,
        'no_caching': False
    }
    return params