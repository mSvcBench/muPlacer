import numpy as np

def Connector(GMA_params):
    # Take params from GMA and return sbmp params
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
    delay_decrease_stop_condition = delay_decrease_target * 0.75 # 75% of the target, TODO export to config
    delay_increase_target = - delay_decrease_target
    delay_increase_stop_condition = delay_increase_target * 0.75 # 75% of the target, TODO export to config

    B = GMA_params['network']['cloud-edge-bps']['value']
    RTT = GMA_params['network']['edge-cloud-rtt']['value']/1000.0
    Cost_cpu_edge = GMA_params['cost']['edge-area']['cpu']['value']
    Cost_mem_edge = GMA_params['cost']['edge-area']['memory']['value']
    Cost_cpu_cloud = GMA_params['cost']['cloud-area']['cpu']['value']
    Cost_mem_cloud = GMA_params['cost']['cloud-area']['memory']['value']
    Cost_network = GMA_params['cost']['cloud-area']['network']['value']

    if 'expanding-depth' in GMA_params['optimizer']['sbmp']:
        expanding_depth = int(GMA_params['optimizer']['sbmp']['expanding-depth'])
    else:
        expanding_depth = 2
    if 'max-sgs' in GMA_params['optimizer']['sbmp']:
        max_sgs = int(GMA_params['optimizer']['sbmp']['max-sgs'])
    else:
        max_sgs = 256
    if 'max-traces' in GMA_params['optimizer']['sbmp']:
        max_traces = int(GMA_params['optimizer']['sbmp']['max-traces'])
    else:
        max_traces = 2048
    if 'traces-b' in GMA_params['optimizer']['sbmp']:
        traces_b = GMA_params['optimizer']['sbmp']['traces-b']
    else:
        traces_b = None
    if 'HPA_cpu_th' in GMA_params['optimizer']['sbmp']:
        HPA_cpu_th = GMA_params['optimizer']['sbmp']['HPA_cpu_th']
    else:
        HPA_cpu_th = None
    if 'locked_b' in GMA_params['optimizer']:
        locked_b = GMA_params['optimizer']['locked']
    else:
        locked_b = np.zeros(M)

    params = {
        'S_edge_b': S_edge_b,
        'Ucpu': Ucpu,
        'Umem': Umem,
        'Fm': Fm,
        'M': M,
        'lambd': lambda_val,
        'L': L,
        'Di': Di,
        'delay_decrease_target': delay_decrease_target,
        'delay_increase_target': delay_increase_target,
        'delay_decrease_stop_condition': delay_decrease_stop_condition,
        'delay_increase_stop_condition': delay_increase_stop_condition,
        'RTT': RTT,
        'B': B,
        'Cost_cpu_edge': Cost_cpu_edge,
        'Cost_mem_edge': Cost_mem_edge,
        'Cost_cpu_cloud': Cost_cpu_cloud,
        'Cost_mem_cloud': Cost_mem_cloud,
        'Cost_network': Cost_network,
        'locked_b': locked_b,
        'Qcpu': Qcpu,
        'Qmem': Qmem,
        'max-sgs': max_sgs,
        'expanding-depth': expanding_depth
    }
    return params