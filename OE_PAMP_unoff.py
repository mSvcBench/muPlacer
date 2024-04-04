import numpy as np
import os
import subprocess
import time
from kubernetes import client, config
from autoplacer_unoffload import autoplacer_unoffload
from build_Fcm import Fcm


#   OE_PAMP function to unoffload microservices from edge cluster to cloud cluster


def OE_PAMP_unoff(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD):

    max_delay_delta = ((SLO_MARGIN_UNOFFLOAD * SLO) - AVG_DELAY) / 1000.0 # Minimum delay delta to satisfy SLO
    #best_S_edge = np.array(eng.autoplacer_unoffload(matlab.double(Rcpu), matlab.double(RMEM), Pcm, M, lambda_value, Rs, app_edge, max_delay_delta)) # Running matlab autoplacer_unoffload
    output = autoplacer_unoffload(RCPU, RMEM, Fcm(prom, PERIOD, APP), M, lambda_value, Rs, APP_EDGE, max_delay_delta, RTT, nargout=2) # Running matlab autoplacer_unoffload
    best_S_edge = np.array(output[0])
    #print("delta_delay:",output[1])
    best_S_edge = np.delete(best_S_edge, -1) # Remove the last value (user) from best_S_edge
    #print(numpy.around(best_S_edge, decimals=0)) # This is the new sequence of microservice that must stay in the edge cluster to reduce the delay
    
    # Reshape the best_S_edge array to match the shape of APP_EDGE if they are different
    if best_S_edge.shape != APP_EDGE.shape:
        # Reshape best_S_edge to match the shape of APP_EDGE
        best_S_edge = np.reshape(best_S_edge, APP_EDGE.shape)
    
    # Get the new microservice to delete in the edge cluster
    if not np.array_equal(best_S_edge, APP_EDGE):
        new_edge = np.subtract(APP_EDGE,best_S_edge) # This is the new microservice to delete from edge cluster
        to_delete = np.array(np.array(APP))[new_edge == 1] # Name of the new microservice to delete from edge cluster
    else:
        print("\rIt's not possible to unoffload any microservice")
        return
    

    ## SCALE DEPLOYMENT IN CLOUD CLUSTER ##
    # Load the kube config file
    config.load_kube_config(context=CTX_CLUSTER2)
    # Create the API object
    api = client.AppsV1Api()
    # Set the same number of replicas in the cloud cluster as the edge cluster
    for name in to_delete:
        name_edge = f"{name}-edge"
        deployment = api.read_namespaced_deployment(name=name_edge, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_cloud = f"{name}-cloud"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_cloud}", "--replicas", str(replicas)], check=True)
    

    ## REMOVE MICROSERVICES FROM EDGE CLUSTER ##
    # Directory where yaml are located
    directory = '/home/alex/Downloads/automate/edge'
    # List to store the matching files
    matching_files = []
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in to_delete):
            # If it does, add it to the list of matching files
            matching_files.append(filename)

    time.sleep(30) # Wait 30 seconds before removing microservices from edge cluster
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "delete", "-f", filepath, "--context", CTX_CLUSTER2], check=True)
    

    ## REMOVE HPAs FROM EDGE CLUSTER ##
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in to_delete):
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "delete", "-f", filepath, "--context", CTX_CLUSTER2], check=True)