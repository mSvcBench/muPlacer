import numpy as np
import os
import subprocess
import time
from kubernetes import client, config
from offload import offload
from build_Fcm import Fcm


#   OE_PAMP function to unoffload microservice instance-sets from edge cluster


def OE_PAMP_unoff(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE):
    max_delay_delta = ((SLO_MARGIN_UNOFFLOAD * SLO) - AVG_DELAY) / 1000.0 # Maximum delay delta to satisfy SLO
    output = offload(RCPU, RMEM, Fcm(prom, PERIOD, APP), M, lambda_value, Rs, APP_EDGE, max_delay_delta, RTT, NE) # Unoffload function
    best_S_edge = np.array(output) # Istance-sets that must stay in the edge cluster according to OE_PAMP
    best_S_edge = np.delete(best_S_edge, -1) # Remove the last value (user) from best_S_edge
    
    # Reshape the best_S_edge array to match the shape of APP_EDGE if they are different
    if best_S_edge.shape != APP_EDGE.shape:
        best_S_edge = np.reshape(best_S_edge, APP_EDGE.shape) # Reshape best_S_edge to match the shape of APP_EDGE
    
    # Get the new microservice to delete in the edge cluster
    if not np.array_equal(best_S_edge, APP_EDGE):
        new_edge = np.subtract(APP_EDGE,best_S_edge) # Microservice instance-sets to delete from edge cluster
        to_delete = np.array(np.array(APP))[new_edge == 1] # Name of microservice instance-sets to delete from edge cluster
    else:
        print("\rIt's not possible to unoffload any microservice")
        return
    

    ## SCALE DEPLOYMENT IN CLOUD CLUSTER ##
    
    config.load_kube_config(context=CTX_CLUSTER2) # Load the kube config file
    api = client.AppsV1Api() # Create the API object
    # Set the same number of replicas in the cloud cluster as the edge cluster
    for name in to_delete:
        name_edge = f"{name}-edge"
        deployment = api.read_namespaced_deployment(name=name_edge, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_cloud = f"{name}-cloud"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_cloud}", "--replicas", str(replicas)], check=True)
    

    ## UNOFFLOADING ##
    directory = MICROSERVICE_DIRECTORY + '/edge' # Directory where manifest files are located
    matching_files = [] # List to store the matching files
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in to_delete):
            matching_files.append(filename) # If it does, add it to the list of matching files

    time.sleep(5) # Wait 5 seconds before removing microservices from edge cluster
    
    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "delete", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command
    

    ## REMOVE HPAs FROM EDGE CLUSTER ##
    matching_files = [] # List to store the matching files
    directory = HPA_DIRECTORY + '/edge' # Directory where HPA yaml files are located

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in to_delete):
            matching_files.append(filename) # If it does, add it to the list of matching files
            
    # Iterate over all files in matching_files
    for filename in matching_files:
        filepath = os.path.join(directory, filename) # Construct the full path to the file
        subprocess.run(["kubectl", "delete", "-f", filepath, "--context", CTX_CLUSTER2], check=True) # Execute the command