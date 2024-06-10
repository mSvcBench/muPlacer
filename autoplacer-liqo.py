import os
import time
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import numpy as np
import subprocess
import numpy
import json
import sys
import threading
import csv
import random


PROMETHEUS_URL = "http://160.80.223.232:30000"
PERIOD = 5 # Rate of queries in minutes
SLO = 80 # Service Level Objective (SLO) in ms
NAMESPACE = "edge" # Namespace of the application
ZONE_LABEL = "edge1" # Label "zone" associated to deployment offloaded
SLO_MARGIN_UNOFFLOAD = 0.8 # SLO increase margin
HPA_STATUS = 0 # Initialization HPA status (0 = HPA Not running, 1 = HPA running)
HPA_MARGIN = 1.05 # HPA margin
RTT = 40e-3 #  Initialization Round Trip Time in seconds
AVG_DELAY = 0 # Average user delay
TRAFFIC = 0 # Cloud-edge traffic
RCPU = np.array([]) # CPU provided to each microservice
RCPU_EDGE = 0 # CPU provided to each microservice in the edge cluster
RCPU_CLOUD = 0 # CPU provided to each microservice in the cloud cluster
SAVE_RESULTS = 0 # Save results in csv file (0 = don't save, 1 = save)
POSITIONING_TYPE = "autoplacer" # Type of positioning strategy used (Autoplacer, Only Cloud, Random, Most Frequently Used, Interaction Aware)
FOLDER = "temporary" # Folder where to save the results

# Connect to Prometheus
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

# Get microservice (app) names of the application
def get_app_names():
    # Get all app names inside the namespace
    command1 = f'kubectl get deployment -n {NAMESPACE} -o custom-columns=APP:.metadata.labels.app --no-headers=true | grep -v "<none>" | sort | uniq'
    result1 = subprocess.run(command1, shell=True, check=True, text=True, stdout=subprocess.PIPE)
    output1 = result1.stdout
    # Split the output by newline, remove any leading/trailing whitespaces, and filter out empty strings
    app_names = [value.strip() for value in output1.split('\n') if value.strip()]
    return app_names

APP_EDGE = np.zeros(len(get_app_names()), dtype=int) # Microservice in the edge cluster

# Function that get the average delay from istio-ingress
def get_avg_delay():
    while True:
        global AVG_DELAY
        query_avg_delay = f'sum by (source_workload) (rate(istio_request_duration_milliseconds_sum{{source_workload="istio-ingress", reporter="source", response_code="200"}}[1m])) / sum by (source_workload) (rate(istio_request_duration_milliseconds_count{{source_workload="istio-ingress", reporter="source", response_code="200"}}[1m]))'
        result_query = prom.custom_query(query=query_avg_delay)
        if result_query:
            for result in result_query:
                avg_delay = result["value"][1]  # avg_delay result
                if avg_delay == "NaN":
                    AVG_DELAY = 0
                    if SAVE_RESULTS == 1:
                        save_delay(AVG_DELAY)
                    time.sleep(1)
                else:
                    AVG_DELAY = round(float(avg_delay),2)
                    if SAVE_RESULTS == 1:
                        save_delay(AVG_DELAY)
                    time.sleep(1)    

# Function that take the requests per second
def get_lamba():
    # Query to obtain requests per second
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{source_app="istio-ingress", reporter="destination", response_code="200"}}[1m]))'
    query_result = prom.custom_query(query=query_lambda)
    if query_result:
        for result in query_result:
            return float(result["value"][1]) # Get the value of lambda

# Function that get the cpu provided to each microservices 
def get_Rcpu():
    global RCPU, RCPU_EDGE, RCPU_CLOUD
    while True:
        app_names = get_app_names() # Get app names with the relative function
        Rcpu_cloud = np.full(len(app_names), -1, dtype=float) # Initialize Rcpu_cloud
        Rcpu_edge = np.full(len(app_names), -1, dtype=float) # Initialize Rcpu_edge
        
        # Query to obtain cpu provided to each microservice in the cloud cluster
        query_cpu_cloud = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="cpu", container!="istio-proxy", node!="{get_liqo_node()}"}})'
        cpu_cloud_results = prom.custom_query(query=query_cpu_cloud)
        if cpu_cloud_results:
            for result in cpu_cloud_results:
                Rcpu_value = result["value"][1] # Get the value of Rcpu
                Rcpu_cloud[app_names.index(result["metric"]["container"])] = float(Rcpu_value) # Insert the value inside Rcpu_cloud array
        
        # Query to obtain cpu provided to each microservice in the edge cluster
        query_cpu_edge = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="cpu", container!="istio-proxy", node="{get_liqo_node()}"}})'
        cpu_edge_results = prom.custom_query(query=query_cpu_edge)
        if cpu_edge_results:
            for result in cpu_edge_results:
                Rcpu_value = result["value"][1] # Get the value of Rcpu
                Rcpu_edge[app_names.index(result["metric"]["container"])] = float(Rcpu_value) # Insert the value inside Rcpu_cloud array
        
        Rcpu_edge_graphs = Rcpu_edge.copy() # Copy Rcpu_edge array to Rcpu_edge_graphs array
        
        # Check for missing values in Rcpu_edge and replace them with corresponding values from Rcpu_cloud
        for i, value in enumerate(Rcpu_edge):
            if value == -1:
                Rcpu_edge[i] = Rcpu_cloud[i]

        # Check for missing values in Rcpu_edge_graphs and replace them with 0 for graphs
        for i, value in enumerate(Rcpu_edge_graphs):
            if value == -1:
                Rcpu_edge_graphs[i] = 0

        # Save Rcpu values of edge and cloud cluster
        RCPU_EDGE = np.sum(Rcpu_edge_graphs)
        RCPU_CLOUD = np.sum(Rcpu_cloud)
        RCPU = np.concatenate((np.append(Rcpu_cloud, 0), np.append(Rcpu_edge, 0))) # Rcpu values of each microservices (cloud+edge)
        time.sleep(5)
    #return Rcpu

# Function that get the memory provided to each microservices
def get_Rmem():
    app_names = get_app_names() # Get app names with the relative function
    Rmem_cloud = np.full(len(app_names), -1, dtype=float) # Initialize Rmem_cloud
    Rmem_edge = np.full(len(app_names), -1, dtype=float) # Initialize Rmem_edge
    
    # Query to obtain cpu provided to each microservice in the cloud cluster
    query_mem_cloud = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="memory", container!="istio-proxy", node!="{get_liqo_node()}"}})'
    mem_cloud_results = prom.custom_query(query=query_mem_cloud)
    if mem_cloud_results:
        for result in mem_cloud_results:
            Rmem_value = result["value"][1] # Get the value of Rmem
            Rmem_cloud[app_names.index(result["metric"]["container"])] = float(Rmem_value) # Insert the value inside Rmem_cloud array
    else:
        Rmem_cloud = np.zeros(len(app_names))

    # Query to obtain cpu provided to each microservice in the edge cluster
    query_mem_edge = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="memory", container!="istio-proxy", node="{get_liqo_node()}"}})'
    mem_edge_results = prom.custom_query(query=query_mem_edge)
    if mem_edge_results:
        for result in mem_edge_results:
            Rmem_value = result["value"][1] # Get the value of Rmem
            Rmem_edge[app_names.index(result["metric"]["container"])] = float(Rmem_value) # Insert the value inside Rmem_cloud array
    else:
        Rmem_cloud = np.zeros(len(app_names))

    # Check for missing values in Rmem_edge and replace them with corresponding values from Rmem_cloud
    for i, value in enumerate(Rmem_edge):
        if value == -1:
            Rmem_edge[i] = Rmem_cloud[i]

    Rmem_edge = np.append(Rmem_edge, 0) # Add user with Rmem = 0
    Rmem_cloud = np.append(Rmem_cloud, 0) # Add user with Rmem = 0
    Rmem = np.concatenate((Rmem_cloud, Rmem_edge)) # Rmem values of each microservices (cloud+edge)
    return Rmem

# Function that get the microservice names already in edge cluster
def get_app_edge():
    global APP_EDGE
    while True:
        #app_edge = np.zeros(len(get_app_names()), dtype=int) # Initialize array with zeros with lenght equal to the number of microservices (the last value is for istio-ingress)
        command = f"kubectl get deployment -n {NAMESPACE} -l zone={ZONE_LABEL} -o custom-columns=APP:.metadata.labels.app --no-headers=true | sort | uniq" # Get microservice offloaded
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
        output = result.stdout
            # Split the output by newline, remove any leading/trailing whitespaces, and filter out empty strings
        app_names_edge = [value.strip() for value in output.split('\n') if value.strip()]
        for microservice in app_names_edge:
            #app_edge[get_app_names().index(microservice)] = 1 # Set value for microservice in edge to 1        
            APP_EDGE[get_app_names().index(microservice)] = 1 # Set value for microservice in edge to 1
        time.sleep(10)
    #return app_edge

# Function that get the node name of edge node
def get_liqo_node():
    command = "kubectl get nodes -o json"
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
    output = result.stdout

    nodes_info = json.loads(output)
    nodes = nodes_info.get("items", [])

    nodes_edge = []

    for node in nodes:
        node_name = node.get("metadata", {}).get("name")
        node_labels = node.get("metadata", {}).get("labels", {})
        if "topology.kubernetes.io/zone" in node_labels and node_labels["topology.kubernetes.io/zone"] == "edge1":
            nodes_edge.append(node_name)

    if len(nodes_edge) != 1:
        error_message = "Error: Multiple agent nodes found."
        print(error_message)
        sys.exit(1)
    
    return nodes_edge[0] if nodes_edge else None

# Function that find probabilities between each microservice including istio-ingress
def get_Pcm():
    app_names = get_app_names() # Get app names with the relative function
    
    # add app name of istio-ingress in app_names list for Pcm matrix
    command = f"kubectl get deployment -n istio-ingress -o custom-columns=APP:.metadata.labels.app --no-headers=true | sort | uniq"
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
    output = result.stdout
    # Split the output by newline, remove any leading/trailing whitespaces, and filter out empty strings
    app_istio_ingress = [value.strip() for value in output.split('\n') if value.strip()]
    #print(app_istio_ingress)
    app_names = app_names + app_istio_ingress # Add istio-ingress to app_names list

    # Create a matrix with numpy to store calling probabilities
    Pcm = np.zeros((len(app_names), len(app_names)), dtype=float)

    # Find and filter significant probabilities
    for i, src_app in enumerate(app_names):
        for j, dst_app in enumerate(app_names):
            # Check if source and destination apps are different
            if src_app != dst_app:
                # total requests that arrive to dst microservice from src microservice
                query1 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[{PERIOD}m]))'
                
                #total requests that arrive to the source microservice
                if src_app != "istio-ingress":
                    # Query for istio-ingress
                    query2 = f'sum by (source_app) (rate(istio_requests_total{{reporter="destination", destination_app="{src_app}",response_code="200"}}[{PERIOD}m]))'
                else:
                    query2 = f'sum by (source_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[{PERIOD}m]))' 

                #print(query1)
                #print(query2)
                r1 = prom.custom_query(query=query1)
                r2 = prom.custom_query(query=query2)

                # Initialize variables
                calling_probability = None
                v = 0
                s = 0

                # Extract values from queries
                if r1 and r2:
                    for result in r1:
                        if float(result["value"][1]) == 0: 
                            continue
                        v = result["value"][1]
                        #print("v =", v)
                    for result in r2:
                        if float(result["value"][1]) == 0:
                            continue
                        s = result["value"][1]
                        #print("s =", s)

                    # Calculate calling probabilities
                    if s == 0 or s == "Nan":
                        calling_probability = 0 # If there isn't traffic
                    else:
                        calling_probability = float(v) / float(s)
                        calling_probability = round(calling_probability, 5)  # Round to 5 decimal places
                        Pcm[i, j] = calling_probability # Insert the value inside Pcm matrix              
    #print(Pcm)
    return Pcm

# Function that take the response size of each microservice
def get_Rs():
    app_names = get_app_names() # Get app names with the relative function

    Rs = np.zeros(len(app_names), dtype=float) # inizialize Rs array

    # app_names combined with OR (|) for prometheus query
    combined_names = "|".join(app_names)

    # Query to obtain response size of each microservice    
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{response_code="200", destination_app=~"{combined_names}", reporter="destination"}}[{PERIOD}m]))/sum by (destination_app) (increase(istio_response_bytes_count{{response_code="200", destination_app=~"{combined_names}", reporter="destination"}}[{PERIOD}m]))'
    r1 = prom.custom_query(query=query_Rs)
    if r1:
        for result in r1:
            Rs_value = result["value"][1]
            if Rs_value == "NaN":
                Rs[app_names.index(result["metric"]["destination_app"])] = 0 # If there isn't traffic
            else:
                Rs[app_names.index(result["metric"]["destination_app"])] = float(Rs_value)
    return Rs

# Autoplacer function to offload
def autoplacer_offload():
    global RTT
    global AVG_DELAY
    global APP_EDGE
    global RCPU
    Rmem = get_Rmem() # Memory provided of each microservice
    Pcm = get_Pcm() # Calling probabilities between each microservice
    Rs = get_Rs() # Response size of each microservice
    M = len(RCPU)/2 # Number of microservices
    lambda_value = get_lamba() # Requests per second
    #app_edge = get_app_edge() # Microservice already in the edge
    min_delay_delta = (AVG_DELAY - SLO) / 1000.0 # Minimum delay delta to satisfy SLO
    #best_S_edge, delta_delay = np.array(eng.autoplacer_offload(matlab.double(Rcpu), matlab.double(Rmem), Pcm, M, lambda_value, Rs, app_edge, min_delay_delta)) # Running matlab autoplacer
    output = eng.autoplacer_offload(matlab.double(RCPU), matlab.double(Rmem), Pcm, M, lambda_value, Rs, APP_EDGE, min_delay_delta, float(RTT), nargout=2) # Running matlab autoplacer
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
        new_edge = numpy.subtract(best_S_edge, APP_EDGE) # This is the new microservice that must stay in the edge cluster to reduce the delay
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

    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)
    

    # SET THE SAME NUMBER OF REPLICAS AS THE CLOUD CLUSTER
    config.load_kube_config() # Load the kube config file

    api = client.AppsV1Api() # Create the API object

    for name in new_edge_names:
        name_cloud = f"{name}-cloud"
        deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_edge = f"{name}-edge"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    

    # APPLY HPA TO MICROSERVICES OFFLOADED
    time.sleep(5)
    # List to store the matching files
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_names):
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)

# Autoplacer function to unoffload
def autoplacer_unoffload():
    global RTT
    global AVG_DELAY
    global RCPU
    Rmem = get_Rmem() # Memory provided of each microservice
    Pcm = get_Pcm() # Calling probabilities between each microservice
    Rs = get_Rs() # Response size of each microservice
    M = len(RCPU)/2 # Number of microservices
    lambda_value = get_lamba() # Requests per second
    #app_edge = get_app_edge() # Microservice already in the edge
    max_delay_delta = ((SLO_MARGIN_UNOFFLOAD * SLO) - AVG_DELAY) / 1000.0 # Minimum delay delta to satisfy SLO
    #best_S_edge = np.array(eng.autoplacer_unoffload(matlab.double(Rcpu), matlab.double(Rmem), Pcm, M, lambda_value, Rs, app_edge, max_delay_delta)) # Running matlab autoplacer_unoffload
    output = eng.autoplacer_unoffload(matlab.double(RCPU), matlab.double(Rmem), Pcm, M, lambda_value, Rs, APP_EDGE, max_delay_delta, float(RTT), nargout=2) # Running matlab autoplacer_unoffload
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
        new_edge = numpy.subtract(APP_EDGE,best_S_edge) # This is the new microservice to delete from edge cluster
        to_delete = np.array(np.array(get_app_names()))[new_edge == 1] # Name of the new microservice to delete from edge cluster
    else:
        print("It's not possible to unoffload any microservice")
        return
    

    ## SCALE DEPLOYMENT IN CLOUD CLUSTER ##
    # Load the kube config file
    config.load_kube_config()
    # Create the API object
    api = client.AppsV1Api()
    # Set the same number of replicas as the cloud cluster
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
        subprocess.run(["kubectl", "delete", "-f", filepath], check=True)
    
    # List to store the matching files
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
        subprocess.run(["kubectl", "delete", "-f", filepath], check=True)

# Random positioning function to offload
def random_positioning():
    global APP_EDGE
    apps = np.ones(len(get_app_names()), dtype=int) 
    not_in_edge = numpy.subtract(apps,APP_EDGE) # # Microservices not in edge cluster
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

    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)
    # Read the cloud deployment
    name_cloud = f"{new_edge_name[0]}-cloud"
    deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)

    # Set the same number of replicas as the cloud cluster
    replicas = deployment.spec.replicas
    if replicas > 0 and replicas != 1:
        # Construct the edge deployment name
        name_edge = f"{new_edge_name[0]}-edge"
        subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    
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
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)

# Most Frequently Used positioning function Used to offload
def mfu_positioning():
    global APP_EDGE
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

    # Iterate over matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)
    
    # Read the cloud deployment
    name_cloud = f"{max_name}-cloud"
    deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)

    # Set the same number of replicas as the cloud cluster
    replicas = deployment.spec.replicas
    if replicas > 0 and replicas != 1:
        # Construct the edge deployment name
        name_edge = f"{max_name}-edge"
        subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    
    
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
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)

# Interaction Aware positioning function to offload
def IA_positioning():
    app_names = sorted(set(get_app_names()) - set(APP_EDGE), key=get_app_names().index) # Microservices not in edge cluster
    # Create a matrix with numpy to store interaction between microservices
    Im = np.zeros((len(app_names), len(app_names)), dtype=float)

    for i, src_app in enumerate(app_names):
        for j, dst_app in enumerate(app_names):
            # Check if source and destination apps are different
            if src_app != dst_app and (APP_EDGE[i] == 0 or APP_EDGE[j] == 0):
                # total requests that arrive to dst microservice from src microservice
                query1 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[{PERIOD}m]))'
                query2 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{dst_app}",reporter="destination", destination_app="{src_app}",response_code="200"}}[{PERIOD}m]))'

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
                interactions = round(interactions, 1)  # Round to 5 decimal places
                Im[i, j] = interactions # Insert the value inside Im matrix              
    
    # Find the maximum value in the interaction matrix
    max_value = Im.max()

    # Find the indices of the maximum values in the interaction matrix
    max_indices = np.argwhere(Im == max_value)
    if len(max_indices) > 1:
        # Sort each index pair to consider [x, y] and [y, x] as equivalent
        sorted_indices = [tuple(sorted(index_pair)) for index_pair in max_indices]
        # Remove duplicates and randomly choose one of the sorted indices
        unique_sorted_indices = list(set(sorted_indices))
        # Randomly choose one pair of microservices (at least one microservice inside the pair only in the cloud cluster)
        chosen_sorted_index = random.choice(unique_sorted_indices)
        # Extract the microservices names with the maximum interaction value
        microservice1 = app_names[chosen_sorted_index[0]]
        microservice2 = app_names[chosen_sorted_index[1]]
    else:
        # If there is only one maximum value, use its indices directly
        chosen_index = max_indices[0]
        microservice1 = app_names[chosen_index[0]]
        microservice2 = app_names[chosen_index[1]]

    # Print or use the values as needed
    #print(f"The maximum interaction value is {max_value} between microservices {microservice1} and {microservice2}.")
    
    # Pair of microservices to offload
    new_edge_names = [microservice1, microservice2]

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

    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)
    

    # SET THE SAME NUMBER OF REPLICAS AS THE CLOUD CLUSTER
    config.load_kube_config() # Load the kube config file

    api = client.AppsV1Api() # Create the API object

    for name in new_edge_names:
        name_cloud = f"{name}-cloud"
        deployment = api.read_namespaced_deployment(name=name_cloud, namespace=NAMESPACE)
        
        replicas = deployment.spec.replicas
        if replicas > 0 and replicas != 1:
            name_edge = f"{name}-edge"
            subprocess.run(["kubectl", "scale", "-n", NAMESPACE, f"deployment/{name_edge}", "--replicas", str(replicas)], check=True)
    
    # APPLY HPA TO MICROSERVICES OFFLOADED
    time.sleep(5)
    # List to store the matching files
    matching_files = []
    # Directory where yaml are located
    directory = '/home/alex/Downloads/hpa/edge'
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Find the yaml file inside the folder
        if any(name in filename for name in new_edge_names):
            # If it does, add it to the list of matching files
            matching_files.append(filename)
    # Iterate over all files in matching_files
    for filename in matching_files:
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        # Execute the command
        subprocess.run(["kubectl", "apply", "-f", filepath], check=True)

# Function that check if some HPA is running
def check_hpa():
    # Load the kube config file
    config.load_kube_config()

    # Create the API object
    api = client.AppsV1Api()
    autoscaling_api = client.AutoscalingV1Api()

    while True:
        global HPA_STATUS
        # Get the list of all deployments in the specified namespace
        deployments = api.list_namespaced_deployment(namespace=NAMESPACE)

        all_hpa_satisfy_condition = True  # Assume initially that all HPAs satisfy the condition

        for deployment in deployments.items:
            deployment_name = deployment.metadata.name

            # Get the list of associated HPAs for the deployment
            associated_hpas = autoscaling_api.list_namespaced_horizontal_pod_autoscaler(namespace=NAMESPACE)
            
            for hpa in associated_hpas.items:
                hpa_name = hpa.metadata.name

                # Check if the HPA is associated with the current deployment
                if hpa.spec.scale_target_ref.name == deployment_name:
                    # Get the target CPU utilization percentage
                    cpu_utilization = hpa.spec.target_cpu_utilization_percentage

                    # Get the current status of the HPA
                    hpa_status = autoscaling_api.read_namespaced_horizontal_pod_autoscaler_status(name=hpa_name, namespace=NAMESPACE)
                    current_cpu_utilization_percentage = hpa_status.status.current_cpu_utilization_percentage

                    if current_cpu_utilization_percentage is None:
                        current_cpu_utilization_percentage = 0

                    # Check if the current CPU usage is less than the target CPU for the HPA
                    if current_cpu_utilization_percentage >= cpu_utilization * HPA_MARGIN:
                        # If an HPA doesn't satisfy the condition, set the flag to False and break out of the loop
                        all_hpa_satisfy_condition = False
                        break

        if all_hpa_satisfy_condition:
            HPA_STATUS = 0
        else:
            HPA_STATUS = 1
        # HPA metrics refresh every 15 seconds
        time.sleep(5)

# Function that get the RTT cloud-edge
def get_RTT():
    global RTT
    while True:
        RTT_array = np.array([])

        # Repeat the RTT measurement 10 times
        for i in range(10):
            # Get the ip of rtt service
            command1 = "kubectl get svc -n edge | awk '/rtt/ {print $3}'"
            result1 = subprocess.run(command1, shell=True, capture_output=True, text=True)
            rtt_svc_ip = result1.stdout.strip()  

            # Get pod name of s1-cloud
            command2 = "kubectl get pods -n edge | awk '/s1-cloud/ {print $1}' | head -n 1"
            result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)
            s1_cloud_pod_name = result2.stdout.strip()  # Pulizia dell'output

            # Get RTT edge-cloud
            RTT_command = f"kubectl exec -n edge -it {s1_cloud_pod_name} -- bash -c 'curl --head -s -w %{{time_total}} http://{rtt_svc_ip} 2>/dev/null | tail -1'"
            RTT_result = subprocess.run(RTT_command, shell=True, capture_output=True, text=True)
            RTT = RTT_result.stdout.strip()  # Pulizia dell'output
            try:
                RTT = float(RTT)  # Conversione in float
            except ValueError:
                continue
            # Add RTT value to the array
            RTT_array = np.append(RTT_array, RTT)
        threshold = 1.5
        # Filter the values
        for i in range (10):
            std_dev = np.std(RTT_array)
            mean = np.mean(RTT_array)
            RTT_array = [x for x in RTT_array if abs(x - mean) < threshold * std_dev]
        rtt = np.mean(RTT_array)
        RTT = round(float(rtt), 4)
        time.sleep(5)

# Function that save the avg delay in csv file
def save_delay(delay_value):
    with open(f'/home/alex/Downloads/matlab/{FOLDER}/{POSITIONING_TYPE}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([delay_value])

# Function that save CPU used in edge and cloud clusters in csv file
def save_Rcpu():
    global RCPU_CLOUD
    global RCPU_EDGE
    while True:
        with open(f'/home/alex/Downloads/matlab/{FOLDER}/cpu_cloud_{POSITIONING_TYPE}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([RCPU_CLOUD])
        with open(f'/home/alex/Downloads/matlab/{FOLDER}/cpu_edge_{POSITIONING_TYPE}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([RCPU_EDGE])
        time.sleep(1)

# Function that get the cloud-edge traffic and save it in csv file
def get_traffic():
    while True:
        global TRAFFIC
        # total requests that arrive to dst microservice from src microservice
        query = f'(sum((rate(istio_response_bytes_sum{{source_workload=~"s.*[0-5]-edge|istio-ingress", destination_workload=~"s.*[0-5]-cloud", reporter="destination", response_code="200"}}[1m])))+sum((rate(istio_request_bytes_sum{{source_workload=~"s.*[0-5]-edge|istio-ingress", destination_workload=~"s.*[0-5]-cloud", reporter="destination", response_code="200"}}[1m])))) *8 /1000000'
        r1 = prom.custom_query(query=query)
        # Extract values from querie
        if r1:
            for result in r1:
                TRAFFIC = round(float(result["value"][1]), 2)
                #print("traffic=",TRAFFIC,"Mbps")
                with open(f'/home/alex/Downloads/matlab/{FOLDER}/{POSITIONING_TYPE}_traffic.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([TRAFFIC])
        time.sleep(1)

def main():
    global HPA_STATUS
    global AVG_DELAY
    stabilization_window_seconds = 120  # window in sec

    # Start the thread that checks the HPAs
    thread_hpa = threading.Thread(target=check_hpa) # Create thread
    thread_hpa.daemon = True # Daemonize thread
    thread_hpa.start()

    # Start the thread that checks the RTT cloud-edge
    thread_RTT = threading.Thread(target=get_RTT) # Create thread
    thread_RTT.daemon = True # Daemonize thread
    thread_RTT.start()
    
    # Start the thread that checks the avg delay
    thread_delay = threading.Thread(target=get_avg_delay) # Create thread
    thread_delay.daemon = True # Daemonize thread
    thread_delay.start()

    # Start the thread that checks microservices in edge cluster
    thread_delay = threading.Thread(target=get_app_edge) # Create thread
    thread_delay.daemon = True # Daemonize thread
    thread_delay.start()

    # Start the thread that checks RCPU in clusters 
    thread_delay = threading.Thread(target=get_Rcpu) # Create thread
    thread_delay.daemon = True # Daemonize thread
    thread_delay.start()

    # Start the thread that checks cloud-edge traffic and save it in csv file
    if SAVE_RESULTS == 1:
        thread_delay = threading.Thread(target=get_traffic) # Create thread
        thread_delay.daemon = True # Daemonize thread
        thread_delay.start()
    
    # Start the thread that save Rcpu in csv file
    if SAVE_RESULTS == 1:
        thread_delay = threading.Thread(target=save_Rcpu) # Create thread
        thread_delay.daemon = True # Daemonize thread
        thread_delay.start()
    
    while True:
        if HPA_STATUS == 0:
            if AVG_DELAY > SLO:
                duration_counter = 0
                while AVG_DELAY > SLO and duration_counter < stabilization_window_seconds and HPA_STATUS == 0:
                    time.sleep(1)
                    if AVG_DELAY > SLO:
                        duration_counter += 1
                    print(f"\r*Current avg_delay: {AVG_DELAY} ms")
                if duration_counter >= stabilization_window_seconds and HPA_STATUS == 0:
                    print(f"\rSLO not satisfied, offloading...")
                    autoplacer_offload()
                    #random_positioning()
                    #mfu_positioning()
                    #IA_positioning()
            elif AVG_DELAY <= SLO*SLO_MARGIN_UNOFFLOAD:
                duration_counter = 0
                while AVG_DELAY < SLO*SLO_MARGIN_UNOFFLOAD and duration_counter < stabilization_window_seconds and HPA_STATUS == 0:
                    time.sleep(1)
                    print(f"\r**Current avg_delay: {AVG_DELAY} ms")
                    duration_counter += 1
                if AVG_DELAY == 0:
                    continue
                elif duration_counter >= stabilization_window_seconds and HPA_STATUS == 0 and APP_EDGE.any() != 0:
                    print(f"\rSLO not satisfied, unoffloading...")
                    autoplacer_unoffload()  
            time.sleep(1)
        else:
            print(f"\rHPA running...")
            time.sleep(1)
if __name__ == "__main__":
    main()