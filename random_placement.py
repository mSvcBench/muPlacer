import numpy as np
import os
import subprocess
import time
from kubernetes import client, config


#   Random function to offload microservices from edge cluster to cloud cluster


def random_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE):
    apps = np.ones(len(APP), dtype=int) 
    not_in_edge = np.subtract(apps,APP_EDGE) # Microservice istance-sets not in edge cluster
    selected = [i for i, element in enumerate(not_in_edge) if element == 1]
    x = np.random.choice(selected, 1) # Randomly select a microservice istance-set to offload
    new_edge = np.zeros(len(APP), dtype=int) # Initialize the new_edge array
    new_edge[x] = 1 # Set the selected microservice istance-set to 1
    new_edge_name = np.array(np.array(APP))[new_edge == 1] # Name of microservice istance-set selected
    
    
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