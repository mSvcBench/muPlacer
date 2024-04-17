import numpy as np
import os
import subprocess
import time
from kubernetes import client, config
from offload import offload
from build_Fcm import Fcm


#   E_PAMP function to offload microservice instance-sets from cloud cluster to edge cluster


def E_PAMP_off(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE):
    min_delay_delta = (AVG_DELAY - SLO) / 1000.0 # Minimum delay delta to satisfy SLO
    output = offload(RCPU, RMEM, Fcm(prom, PERIOD, APP), int(M), lambda_value, Rs, APP_EDGE, min_delay_delta, RTT, NE) # Offload function
    best_S_edge = np.array(output) # Istance-sets that must stay in the edge cluster according to E_PAMP
    best_S_edge = np.delete(best_S_edge, -1) # Remove the last value (user) from best_S_edge
    
    # Reshape the best_S_edge array to match the shape of APP_EDGE if they are different
    if best_S_edge.shape != APP_EDGE.shape:
        best_S_edge = np.reshape(best_S_edge, APP_EDGE.shape) # Reshape best_S_edge to match the shape of APP_EDGE
    
    # Check if best_S_edge is different from APP_EDGE
    if not np.array_equal(best_S_edge, APP_EDGE):
        new_edge = np.subtract(best_S_edge, APP_EDGE) 
    new_edge_names = np.array(np.array(APP))[new_edge == 1] # Microsrvice instance-sets to offload to the edge cluster
    
    
    ## OFFLOADING TO EDGE CLUSTER ##
    directory = MICROSERVICE_DIRECTORY + '/edge' # Directory where manifest files are located
    matching_files = [] # Define a list to store matching files

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find yaml files inside the folder
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
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE,"--context",CTX_CLUSTER2, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    

    ## APPLY HPA TO ISTANCE-SETS OFFLOADED ##
    time.sleep(5)
    matching_files = [] # List to store the matching files
    directory = HPA_DIRECTORY + '/edge' # Directory where HPA yaml files are located

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(f"{name}-" in filename for name in new_edge_names):
            matching_files.append(filename) # If it does, add it to the list of matching files

    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command