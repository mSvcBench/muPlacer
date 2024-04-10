import numpy as np
import os
import subprocess
import time
import random
from kubernetes import client, config


#   Interaction Aware (IA) function to offload microservices istance-sets from cloud cluster to edge cluster


def IA_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE):

    app_names = sorted(set(APP) - set(APP_EDGE), key=APP.index) # Microservices not in edge cluster
    Im = np.zeros((len(app_names), len(app_names)), dtype=float) # Create a matrix to store interaction between microservice istance-sets

    # Construct the interaction matrix Im
    for i, src_app in enumerate(app_names):
        for j, dst_app in enumerate(app_names):
            # Check if source and destination istance-sets are different
            if src_app != dst_app and (APP_EDGE[i] == 0 or APP_EDGE[j] == 0):
                # total requests that arrive to dst microservice from src microservice
                query1 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[2m]))'
                query2 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{dst_app}",reporter="destination", destination_app="{src_app}",response_code="200"}}[2m]))'

                r1 = prom.custom_query(query=query1)
                r2 = prom.custom_query(query=query2)

                # Initialize variables
                interactions = None
                v = 0
                s = 0

                # Extract values from queries
                if r1:
                    for result in r1:
                        if float(result["value"][1]) == 0: 
                            continue
                        v = result["value"][1]

                if r2:    
                    for result in r2:
                        if float(result["value"][1]) == 0:
                            continue
                        s = result["value"][1]

                interactions = (float(v) + float(s)) /2
                interactions = round(interactions, 1)  # Round decimal places
                Im[i, j] = interactions # Insert the value inside Im matrix              
    
    # Find the maximum value in the interaction matrix
    max_value = Im.max()

    # Find the indices of the maximum values in the interaction matrix
    max_indices = np.argwhere(Im == max_value)

    if len(max_indices) > 1:
        sorted_indices = [tuple(sorted(index_pair)) for index_pair in max_indices] # Sort each index pair to consider [x, y] and [y, x] as equivalent
        unique_sorted_indices = list(set(sorted_indices)) # Remove duplicates and randomly choose one of the sorted indices
        chosen_sorted_index = random.choice(unique_sorted_indices) # Randomly choose one pair of microservices (at least one microservice inside the pair only in the cloud cluster)
        # Extract the microservice istance-set with the maximum interaction value
        microservice1 = app_names[chosen_sorted_index[0]]
        microservice2 = app_names[chosen_sorted_index[1]]
    else:
        # If there is only one maximum value, use its indices directly
        chosen_index = max_indices[0]
        microservice1 = app_names[chosen_index[0]]
        microservice2 = app_names[chosen_index[1]]
    
    # Pair of microservice istance-set to offload
    new_edge_names = [microservice1, microservice2]


    ## OFFLOADING TO EDGE CLUSTER ##
    directory = MICROSERVICE_DIRECTORY + '/edge' # Directory where manifest files are located
    matching_files = [] # Define a list to store matching files

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_names):
            matching_files.append(filename) # If it does, add it to the list of matching files

    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command
    

    ## SET THE SAME NUMBER OF REPLICAS AS THE CLOUD CLUSTER ##
    config.load_kube_config() # Load the kube config file
    api = client.AppsV1Api() # Create the API object

    for name in new_edge_names:
        name_cloud = f"{name}-cloud"
        deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_edge = f"{name}-edge"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas), "--context", CTX_CLUSTER2], check=True)
    

    ## APPLY HPAs TO ISTANCE-SETS OFFLOADED ##
    time.sleep(5)
    matching_files = [] # List to store the matching files
    directory = HPA_DIRECTORY + '/edge' # Directory where HPA yaml files are located
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_names):
            matching_files.append(filename) # If it does, add it to the list of matching files
    
    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command