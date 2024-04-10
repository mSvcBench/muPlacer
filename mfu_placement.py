import numpy as np
import os
import subprocess
import time
import random
from kubernetes import client, config

#   Most Frequently Used (MFU) function to offload microservice istance-sets from cloud cluster to edge cluster

def mfu_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE):
    apps = np.ones(len(APP), dtype=int) 
    not_in_edge = np.subtract(apps,APP_EDGE) # Microservice istance-sets not in edge cluster
    names = np.array(np.array(APP))[not_in_edge == 1] # Name of microservice istance-sets selected
    combined_names = "|".join(names) # Combine the names of the istance-sets

    # Query to get the most frequently used microservice istance-set
    query_lambda = f'topk(1,sum by (destination_app) (floor(rate(istio_requests_total{{destination_app=~"{combined_names}", reporter="destination", response_code="200"}}[1m]))))'
    query_result = prom.custom_query(query=query_lambda)
    if query_result:
        random_result = random.choice(query_result)
        new_edge_name = random_result['metric']['destination_app'] # Name of the most frequently used microservice istance-set

     ## OFFLOADING TO EDGE CLUSTER ##
    directory = MICROSERVICE_DIRECTORY + '/edge' # Directory where manifest files are located
    matching_files = [] # Define a list to store matching files

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find yaml files inside the folder
        if any(name in filename for name in new_edge_name):
            matching_files.append(filename) # If it does, add it to the list of matching files

    # Iterate over all files in matching_files
    for filename in matching_files: 
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command
    

    ## SET THE SAME NUMBER OF REPLICAS AS THE CLOUD CLUSTER ##
    config.load_kube_config() # Load the kube config file
    api = client.AppsV1Api() # Create the API object

    for name in new_edge_name:
        name_cloud = f"{name}-cloud"
        deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_edge = f"{name}-edge"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE,"--context",CTX_CLUSTER2, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    

    ## APPLY HPA TO ISTANCE-SETS OFFLOADED ##
    time.sleep(5)
    matching_files = [] # List to store the matching files
    directory = HPA_DIRECTORY + '/edge' # Directory where HPA yaml files are located

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(f"{name}-" in filename for name in new_edge_name):
            matching_files.append(filename) # If it does, add it to the list of matching files

    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command