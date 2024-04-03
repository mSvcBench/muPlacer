import numpy as np
import os
import subprocess
import time
from kubernetes import client, config
from autoplacer_offload import autoplacer_offload
from muPlacer import get_app_names
from build_Fcm import Fcm


#   OE_PAMP function to unoffload microservices from edge cluster to cloud cluster

def OE_PAMP_off(RTT, AVG_DELAY, APP_EDGE, RCPU, Rmem, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD):

    min_delay_delta = (AVG_DELAY - SLO) / 1000.0 # Minimum delay delta to satisfy SLO
    #best_S_edge, delta_delay = np.array(autoplacer_offload(Rcpu, Rmem, Pcm, M, lambda_value, Rs, app_edge, min_delay_delta)) # Running matlab autoplacer
    output = autoplacer_offload(RCPU, Rmem, Fcm(prom, PERIOD), int(M), lambda_value, Rs, APP_EDGE, min_delay_delta, RTT) # Running matlab autoplacer
    best_S_edge = np.array(output[0])
    #print("delta_delay:",output[1])
    best_S_edge = np.delete(best_S_edge, -1) # Remove the last value (user) from best_S_edge
    #print(numpy.around(best_S_edge, decimals=0)) # This is the new sequence of microservice that must stay in the edge cluster to reduce the delay
    
    # Reshape the best_S_edge array to match the shape of APP_EDGE if they are different
    if best_S_edge.shape != APP_EDGE.shape:
        # Reshape best_S_edge to match the shape of APP_EDGE
        best_S_edge = np.reshape(best_S_edge, APP_EDGE.shape)
    
    # Get the new microservice that must stay in the edge cluster
    if not np.array_equal(best_S_edge, APP_EDGE):
        new_edge = np.subtract(best_S_edge, APP_EDGE) # This is the new microservice that must stay in the edge cluster to reduce the delay
    new_edge_names = np.array(np.array(get_app_names()))[new_edge == 1] # New microsrvices that must stay in the edge cluster according to PAMP algorithm
    
    # Directory where yaml are located
    directory = '/home/alex/Downloads/automate/edge'

    # List to store the matching files
    matching_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_names):
            # If it does, add it to the list of matching files
            matching_files.append(filename)

    ## OFFLOADING MICROSERVICES TO EDGE CLUSTER ##
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)
    

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
    

    ## APPLY HPA TO MICROSERVICES OFFLOADED ##
    time.sleep(5)
    # List to store the matching files
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(f"{name}-" in filename for name in new_edge_names):
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)