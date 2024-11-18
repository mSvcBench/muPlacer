import numpy as np

def Connector(GMA_params):
    # Take params from GMA and return EPAMP params
    M = GMA_params['n-services']
    S_edge_b = np.minimum(GMA_params['hpa']['edge-area']['current-replicas'],1)
    S_edge_b[M-1]=1 # Last service is always on the edge since it represent the istio-ingress/user
    Ucpu = np.zeros(2*M)
    Ucpu[0:M] = GMA_params['ucpu']['cloud-area']['value']
    Ucpu[M:2*M] = GMA_params['ucpu']['edge-area']['value']
    Umem = np.zeros(2*M)
    Umem[0:M] = GMA_params['umem']['cloud-area']['value']
    Umem[M:2*M] = GMA_params['umem']['edge-area']['value']
    Qcpu = np.zeros(2*M)
    Qcpu[0:M] = GMA_params['qcpu']['cloud-area']['value']
    Qcpu[M:2*M] = GMA_params['qcpu']['edge-area']['value']
    Qmem = np.zeros(2*M)
    Qmem[0:M] = GMA_params['qmem']['cloud-area']['value']
    Qmem[M:2*M] = GMA_params['qmem']['edge-area']['value']
    Fm = GMA_params['fm']['value']
    lambda_val = GMA_params['service-lambda']['value'][M-1]
    L = GMA_params['response-length']['value']
    if 'di' in GMA_params:
        Di = GMA_params['di']['value']
    else:
        Di = np.zeros(2*M)
    delay_decrease_target = (GMA_params['edge-user-delay']['value'] - GMA_params['edge-user-target-delay']['value'])/1000.0
    delay_increase_target = - delay_decrease_target
    B = GMA_params['network']['cloud-edge-bps']['value']
    RTT = GMA_params['network']['edge-cloud-rtt']['value']/1000.0
    Cost_cpu_edge = GMA_params['cost']['edge-area']['cpu']['value']
    Cost_mem_edge = GMA_params['cost']['edge-area']['memory']['value']

    if 'expanding-depth' in GMA_params['optimizer']['epamp']:
        expanding_depth = int(GMA_params['optimizer']['epamp']['expanding-depth'])
    else:
        expanding_depth = 2
    if 'max-sgs' in GMA_params['optimizer']['epamp']:
        max_sgs = int(GMA_params['optimizer']['epamp']['max-sgs'])
    else:
        max_sgs = 64
    if 'locked' in GMA_params['optimizer']:
        locked = GMA_params['optimizer']['locked']
    else:
        locked = None

    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Ucpu,
        'Amem': Umem,
        'Fcm': Fm,
        'M': M,
        'lambd': lambda_val,
        'Rs': L,
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'delay_increase_target': delay_increase_target,
        'RTT': RTT,
        'Ne': B,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'locked': locked,
        'dependency_paths_b': None,
        'u_limit': u_limit,
        'Qcpu': Qcpu,
        'Qmem': Qmem
    }
    return params