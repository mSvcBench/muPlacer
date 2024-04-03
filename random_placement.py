from muPlacer import get_app_names
import numpy as np
import os
import subprocess
import time
import random
from kubernetes import client, config

#   Random function to offload microservices from edge cluster to cloud cluster


def random_placement(RTT, AVG_DELAY, APP_EDGE, RCPU, Rmem, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD):
    apps = np.ones(len(get_app_names()), dtype=int) 
    not_in_edge = np.subtract(apps,APP_EDGE) # # Microservices not in edge cluster
    selected = [i for i, element in enumerate(not_in_edge) if element == 1]
    x = np.random.choice(selected, 1)
    new_edge = np.zeros(len(get_app_names()), dtype=int)
    new_edge[x] = 1
    new_edge_name = np.array(np.array(get_app_names()))[new_edge == 1] # Name of microservice selected
    # Directory where yaml are located
    directory = '/home/alex/Downloads/automate/edge'

    # List to store the matching files
    matching_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_name):
            # If it does, add it to the list of matching files
            matching_files.append(filename)

    # Load the kube config file
    config.load_kube_config()

    # Create the API object
    api = client.AppsV1Api()

    ## OFFLOADING A RANDOM MICROSERVICE TO EDGE CLUSTER ##
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)
    
    # APPLY HPA TO MICROSERVICES OFFLOADED
    time.sleep(5)
    # List to store the matching files
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_name):
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)

    
    ## SET THE SAME NUMBER OF REPLICAS AS THE CLOUD CLUSTER ##
    # Read the cloud deployment
    name_cloud = f"{new_edge_name[0]}-cloud"
    deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)

    # Set the same number of replicas as the cloud cluster
    replicas = deployment.spec.replicas
    if replicas > 0 and replicas != 1:
        # Construct the edge deployment name
        name_edge = f"{new_edge_name[0]}-edge"
        subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas), "--context", CTX_CLUSTER2], check=True)
