import numpy as np
import networkx as nx
from S2id import S2id
from old.delayMat import delayMat
from id2S import id2S


#   RTT : RTT cloud edge
#   Ne : cloud edge bit rate
#   lambda : user request frequency
#   Rs : byte lenght of the response of microservices
#   Fcm : call frequency matrix
#   Nc : number of time a microservice is called per request
#   M : number of microservices
#   Rcpu_req : CPU seconds for internal functions
#   Ce, Me, Ne : CPU, Mem and net capacity of the edge
#   Rcpu, Rmem : CPU and Mem requested by microservices
#   db = 0 if no db
#   e : number of datacenters
#   app_edge: microservices already at the edge cloud
#   min_delay_delta: minimum delay reduction

# _b : binary encoding of a set of nodes/services 
# _id : identifier of a set of nodes come out from binary encoding
# _n : identifiers of the nodes of a set

def heuristic_unoffload(Fcm, RTT, Rcpu_req, Rcpu, Rmem, Ce, Me, Ne, lambd, Rs, M, db, horizon, e, app_edge, max_delay_delta):
    ## SEARCH ALL PATHS FROM USER TO INSTANCES ##
    G = nx.DiGraph(Fcm) # Create the graph of the mesh with probabilities
    user = G.number_of_nodes() # User is the last microservice instance-set (root in the graph)
    last_n = user # User is the root of the graph
    paths_n = [] # Define the variable for dependency paths


    ## FIND ALL PATHS WITH THE LEAF MICROSERVICE IN CLOUD CLUSTER ("valid" path) ##    
    for s in range(1, last_n):
        # Find all paths in the graph
        paths = list(nx.all_simple_paths(G, source=user-1, target=s-1))
        # define a variable for "valid" path
        valid_path = True 
        # Check if the path is "valid"
        for path in paths:
            app_edge_values = app_edge[path]
            # If any microservices in the path have app_edge_values == 0, this path is not "valid"
            if any(app_edge_values == 0):
                valid_path = False
                break
        
        # Add the path if it's "valid" in paths_n
        if valid_path:
            paths_n.extend(paths)

    # add path with no microservice at edge
    paths_n.extend([[user-1]])

    ## CREATE THE LIST OF POSSIBLE ID SUBGRAPHS ##
    subgraphs_id_origin = [S2id(app_edge)] # The origin subgraph is the actual configuration running
    # Next cycle will increase it adding possible subgraphs
    for i in range(len(paths_n)):
        SG = paths_n[i]  # SG instances combination of i-th path
        subgraph_b = np.zeros(user)  # Inizialize the subgraph_b vector
        subgraph_b[SG] = 1  # Assign value 1 in subgraph_b vector facing subgraph_n
        subgraph_id = S2id(subgraph_b)  # Convert the subgraph in subgraph_id
        # Check if there is already the current subgraph_id in the list
        if subgraph_id not in subgraphs_id_origin:
            subgraphs_id_origin.append(subgraph_id)  # Add the current subgraph in the id list


    ## GREEDY SUBTRACTION OF SUBGRAPHS FROM EDGE CLUSTER##
    best_edge_Sid = [] 
    for h in range(2, e+1):
        # Repeat the algorithm for every edge data center, considering in each iteration a subproblem made by only a cloud and an edge
        subgraphs_id = subgraphs_id_origin.copy()
        Scur_edge_origin_id = subgraphs_id[0] # Current state id
        H = [] # History vector
        Rcpu_edge = np.array(Rcpu[(h - 1) * M:h * M]) # CPU requested by instances in the edge
        Rmem_edge = np.array(Rmem[(h - 1) * M:h * M]) # Memory requested by instances in the edge
        Scur_edge_origin_b = np.zeros(M)
        Rcpu_origin = np.sum(Scur_edge_origin_b * Rcpu_edge) # Total CPU requested by instances in the edge
        Rmem_origin = np.sum(Scur_edge_origin_b * Rmem_edge) # Total Memory requested by instances in the edge
        
        # Build the state vector of starting configuration
        Sorigin = np.zeros(2*M)
        Sorigin[:M-1] = 1
        Sorigin[M:2*M] = app_edge
        
        dorigin = delayMat(Sorigin, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the original configuration
        Scur_edge_id = 2 # Starting configuration with no instances in the edge cluster (only the user in the edge cluster)
        while True:
            nsg = len(subgraphs_id) # Number of subgraphs
            subgraphs_weights = -np.inf * np.ones(nsg) # Initialize the weight array of the subgraphs
            subgraphs_costs = -np.inf * np.ones(nsg) # Initialize the cost array of the subgraphs
            subgraphs_r = np.inf * np.ones(nsg) # Initialize the delay reduction array of the subgraphs
            subgraph_d = np.inf * np.ones(nsg) # Initialize the new delay variable array of the subgraphs

            # For each configuration calculate delay reduction, cost, weight and new delay
            for i in range(nsg):
                sg_id = subgraphs_id[i] # Current subgraph id
                Snew_edge_id = np.bitwise_or(sg_id-1, Scur_edge_id-1) + 1 # New edge state adding new subgraph
                Snew_edge_b = id2S(Snew_edge_id, 2**M) # New edge state in binary encoding
                Rcpu_new = np.sum(Snew_edge_b * Rcpu_edge) # CPU requested by the new state
                Rmem_new = np.sum(Snew_edge_b * Rmem_edge) # Memory requested by the new state
                
                # Check if the new configuration is feasible (resource exhaustion)
                if Rcpu_new > Ce or Rmem_new > Me:
                    subgraphs_weights[i] = -np.inf
                    subgraphs_r[i] = 0
                    continue
                
                cost_cpu = Rcpu_new - Rcpu_origin + 1e-6 # CPU cost
                cost_mem = Rmem_new - Rmem_origin + 1e-6 # Memory cost
                cost = cost_cpu # Cost of the new state
                Snew = np.zeros(2*M) # Inizialize array of the new state
                Snew[:M-1] = 1 # Set the cloud instances in the array (all instances always in the cloud cluster)
                Snew[M:] = Snew_edge_b # Set the edge instances in the array
                dnew = delayMat(Snew, Fcm, Rcpu, Rcpu_req, RTT, Ne, lambd, Rs, M, 2) # Delay of the new state
                
                # Check if the new configuration is feasible about the delay
                if dnew == np.inf:
                    subgraphs_weights[i] = -np.inf
                    subgraphs_r[i] = 0
                    continue

                r = dnew - dorigin # Delay increase of the new configuration
                
                # Check if the new configuration is feasible about delay reduction and set the weight, cost and delay reduction of the new configuration
                if r > max_delay_delta:
                    subgraphs_weights[i] = 0
                    subgraphs_costs[i] = np.inf
                else:
                    subgraphs_weights[i] = min(r, max_delay_delta) / cost  # Weight of the new state (min between delay reduction and max_delay_delta divided by cost)
                    subgraphs_costs[i] = cost # Cost of the new state
                subgraphs_r[i] = r # Delay reduction of the new state
                subgraph_d[i] = dnew # Delay of the new state

            I = np.argmax(subgraphs_weights) # Select the best subgraph
            best_sg = subgraphs_id[I] # Best subgraph id
            Scur_edge_id = np.bitwise_or(best_sg-1, Scur_edge_id-1) + 1  # Update edge status inserting the nodes of the best subgraph
            H.append(np.array([Scur_edge_id, subgraphs_r[I], subgraphs_costs[I], subgraphs_weights[I], subgraph_d[I]])) # Add the best subgraph in the history vector with its properties

            # Prune not considered subgraphs whose nodes are already contained in the edge  
            PR = []
            for pr in range(nsg):
                if np.bitwise_and(subgraphs_id[pr]-1, Scur_edge_id-1) + 1 == subgraphs_id[pr]:
                    PR.append(pr)
            subgraphs_id = [subgraphs_id[pr] for pr in range(nsg) if pr not in PR]
            if len(subgraphs_id) == 0:
                # All subgraphs considered
                break

        # Check if there are solutions
        if len(H) == 0:
            best_edge_Sid[h] = Scur_edge_origin_id # If there are no solutions the best configuration is the original one
        else:
            cur_delay = H[-1][4] # current delay (last solution in H will contain the current edge configuration)
            # Select in H matrix the configuration in which the increment of the delay respect to cur_delay is less than max_delay_delta
            I = [i for i in range(len(H)) if H[i][4] - cur_delay < max_delay_delta]
            # Select the one with less usage of CPU
            I2 = np.argmin([H[i][2] for i in I])

            # If there are no solutions the best configuration is the current one, otherwise the best configuration is appended in the best_edge_Sid vector
            if len(I) == 0:
                best_edge_Sid.append(H[-1][0])
            else:
                best_edge_Sid.append(H[I[I2]][0])

    ## BUILD THE SELECTED CONFIGURATION ARRAY ##
    best_S = np.zeros(e*M) # Initialize the best configuration vector
    best_S[:M-1] = 1 # Set the cloud instances in the array (all instances always in the cloud cluster)
    for h in range(2, e+1):
        best_S[(h-1)*M:h*M] = id2S(int(best_edge_Sid[h-2]), 2**M) # Set the edge instances in the array

    return best_S