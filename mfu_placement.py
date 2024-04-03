from muPlacer import get_app_names
import numpy as np
import os
import subprocess
import time
import random
from kubernetes import client, config

#   Most Frequently Used function to offload microservices from edge cluster to cloud cluster

def mfu_placement(RTT, AVG_DELAY, APP_EDGE, RCPU, Rmem, Pcm, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom):
    apps = np.ones(len(get_app_names()), dtype=int) 
    not_in_edge = np.subtract(apps,APP_EDGE) # This is the new microservice that must stay in the edge cluster to reduce the delay
    new_edge_names = np.array(np.array(get_app_names()))[not_in_edge == 1] # Microservice not in edge cluster
    combined_names = "|".join(new_edge_names)
    query_lambda = f'topk(1,sum by (destination_app) (floor(rate(istio_requests_total{{destination_app=~"{combined_names}", reporter="destination", response_code="200"}}[1m]))))'
    query_result = prom.custom_query(query=query_lambda)
    if query_result:
        random_result = random.choice(query_result)
        max_name = random_result['metric']['destination_app']
    print(max_name)
    # Directory where yaml files are located
    directory = '/home/alex/Downloads/automate/edge'

    # List to store matching files
    matching_files = [filename for filename in os.listdir(directory) if max_name in filename]
    
    # Load the kube config file
    config.load_kube_config()

    # Create the API object
    api = client.AppsV1Api()

    ## OFFLOADING THE MFU MICROSERVICE TO EDGE CLUSTER ##
    # Iterate over matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)
    
    # Read the cloud deployment
    name_cloud = f"{max_name}-cloud"
    deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)

    # Set the same number of replicas as the cloud cluster
    replicas = deployment.spec.replicas
    if replicas > 0 and replicas != 1:
        # Construct the edge deployment name
        name_edge = f"{max_name}-edge"
        subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas), "--context", CTX_CLUSTER2], check=True)
    
    
    # APPLY HPA TO MICROSERVICES OFFLOADED
    time.sleep(5)
    # List to store the matching files
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if max_name in filename:
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath, "--context", CTX_CLUSTER2], check=True)
