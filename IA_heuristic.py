import numpy as np
import utils
from computeNc import computeNc
from buildFci import buildFci
from computeDTot import computeDTot
from numpy import inf


def IA_heuristic(params):

    S_edge_old = params['S_edge_b']
    Acpu_old = params['Acpu']
    Amem_old = params['Amem']
    Fcm = params['Fcm']
    M = params['M']
    lambd = params['lambd']
    Rs = params['Rs']
    Di = params['Di']
    delay_decrease_target = params['delay_decrease_target']
    RTT = params['RTT']
    Ne = params['Ne']
    Cost_cpu_edge = params['Cost_cpu_edge']
    Cost_mem_edge = params['Cost_mem_edge']

    Qmem = params['Qmem'] if 'Qmem' in params else np.zeros(2*M)
    Qcpu = params['Qcpu'] if 'Qcpu' in params else np.zeros(2*M)

    Rs = np.tile(Rs, 2)  # Expand the Rs vector to to include edge and cloud
    S_b_old = np.concatenate((np.ones(int(M)), S_edge_old))
    S_b_old[M-1] = 0  # User is not in the cloud
    Cost_edge_old = utils.computeCost(Acpu_old[M:], Amem_old[M:], Qcpu[M:], Qmem[M:] ,Cost_cpu_edge, Cost_mem_edge)[0] # Total cost of old state
    
    ## COMPUTE THE DELAY OF THE OLD STATE ##
    Fci_old = np.matrix(buildFci(S_b_old, Fcm, M))
    Nci_old = computeNc(Fci_old, M, 2)
    delay_old = computeDTot(S_b_old, Nci_old, Fci_old, Di, Rs, RTT, Ne, lambd, M)[0]
    Nc = computeNc(Fcm, M, 1)
    delay_decrease_new = 0
    S_b_new = S_b_old.copy()


    # DEFINE DICTIONARY FOR INTERACTION AWARE MATRIX ##
    maxes = {
        "ms_i": 0,
        "ms_j": 0,
        "interaction_freq": -1
    }

    if delay_decrease_target > 0:
        ## OFFLOAD ##
        while delay_decrease_target > delay_decrease_new:
            maxes["interaction_freq"] = -1
            maxes["ms_i"] = 0
            maxes["ms_j"] = 0

            ## FIND THE COUPLE OF  MICROSERVICES WITH THE MOST INTERACTIONS ##
            for i in range (M):
                for j in range (M):
                    if i==j:
                        continue
                    if S_b_new[i+M]==0 or S_b_new[j+M]==0:
                        x = Nc[i] * Fcm[i,j] +  Nc[j] * Fcm[j,i]
                        if x > maxes["interaction_freq"]:
                            maxes["interaction_freq"] = x
                            maxes["ms_i"] = i
                            maxes["ms_j"] = j

            S_b_new[maxes["ms_i"]+M] = 1
            S_b_new[maxes["ms_j"]+M] = 1
            
            Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
            Nci_new = computeNc(Fci_new, M, 2)
            delay_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)[0] 
            delay_decrease_new = delay_old - delay_new
            if np.all(S_b_new[M:] == 1):
                # all instances at the edge
                break
    
    ## UNOFFLOAD  ##
    else:
        print("ToDo")

    # compute final values
    Acpu_new = np.zeros(2*M)
    Amem_new = np.zeros(2*M)
    Fci_new = np.matrix(buildFci(S_b_new, Fcm, M))
    Nci_new = computeNc(Fci_new, M, 2)
    delay_new,di_new,dn_new,rhoce_new = computeDTot(S_b_new, Nci_new, Fci_new, Di, Rs, RTT, Ne, lambd, M)
    delay_decrease_new = delay_old - delay_new
    utils.computeResourceShift(Acpu_new, Amem_new, Nci_new, Acpu_old, Amem_old, Nci_old)
    Cost_edge_new = utils.computeCost(Acpu_new[M:], Amem_new[M:], Qcpu[M:], Qmem[M:], Cost_cpu_edge, Cost_mem_edge)[0] # Total cost of new state
    cost_increase_new = Cost_edge_new - Cost_edge_old 

    result = dict()
    result['S_edge_b'] = S_b_new[M:].astype(int)
    result['Cost'] = Cost_edge_new
    result['delay_decrease'] = delay_decrease_new
    result['cost_increase'] = cost_increase_new
    result['Acpu'] = Acpu_new
    result['Amem'] = Amem_new
    result['Fci'] = Fci_new
    result['Nci'] = Nci_new
    result['delay'] = delay_new
    result['di'] = di_new
    result['dn'] = dn_new
    result['rhoce'] = rhoce_new

    return result


        
