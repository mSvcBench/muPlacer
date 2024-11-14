import numpy as np

def Connector(GMA_params):
    # Take params from GMA and return EPAMP params
    M = GMA_params['n-services']
    S_edge_b = np.minimum(GMA_params['hpa']['edge-area']['current-replicas'],1)
    S_edge_b[M-1]=1 # Last service is always on the edge since it represent the istio-ingress/user
    Acpu = np.zeros(2*M)
    Acpu[0:M] = GMA_params['acpu']['cloud-area']['value']
    Acpu[M:2*M] = GMA_params['acpu']['edge-area']['value']
    Amem = np.zeros(2*M)
    Amem[0:M] = GMA_params['amem']['cloud-area']['value']
    Amem[M:2*M] = GMA_params['amem']['edge-area']['value']
    Qcpu = np.zeros(2*M)
    Qcpu[0:M] = GMA_params['qcpu']['cloud-area']['value']
    Qcpu[M:2*M] = GMA_params['qcpu']['edge-area']['value']
    Qmem = np.zeros(2*M)
    Qmem[0:M] = GMA_params['qmem']['cloud-area']['value']
    Qmem[M:2*M] = GMA_params['qmem']['edge-area']['value']
    Fcm = GMA_params['fcm']['value']
    lambda_val = GMA_params['service-lambda']['value'][M-1]
    Rs = GMA_params['rs']['value']
    if 'di' in GMA_params:
        Di = GMA_params['di']['value']
    else:
        Di = np.zeros(2*M)
    delay_decrease_target = (GMA_params['edge-user-delay']['value'] - GMA_params['edge-user-target-delay']['value'])/1000.0
    delay_increase_target = - delay_decrease_target
    Ne = GMA_params['network']['cloud-edge-bps']['value']
    RTT = GMA_params['network']['edge-cloud-rtt']['value']/1000.0
    Cost_cpu_edge = GMA_params['cost']['edge-area']['cpu']['value']
    Cost_mem_edge = GMA_params['cost']['edge-area']['memory']['value']

    if 'u-limit' in GMA_params['optimizer']['epamp']:
        u_limit = int(GMA_params['optimizer']['epamp']['u-limit'])
    else:
        u_limit = 2
    if 'locked' in GMA_params['optimizer']:
        locked = GMA_params['optimizer']['locked']
    else:
        locked = None

    params = {
        'S_edge_b': S_edge_b,
        'Acpu': Acpu,
        'Amem': Amem,
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
        'locked': locked,
        'dependency_paths_b': None,
        'u_limit': u_limit,
        'Qcpu': Qcpu,
        'Qmem': Qmem
    }
    return params