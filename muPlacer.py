import time
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import numpy as np
import subprocess
import threading
import csv
from conf import *
from OE_PAMP_unoff import OE_PAMP_unoff
from OE_PAMP_off import OE_PAMP_off
from IA_placement import IA_placement
from random_placement import random_placement
from mfu_placement import mfu_placement


PERIOD = 3 # Rate of queries in minutes
#SLO_MARGIN_OFFLOAD = 0.9 # SLO increase margin
HPA_STATUS = 0 # Initialization HPA status (0 = HPA Not running, 1 = HPA running)
HPA_MARGIN = 1.05 # HPA margin
RTT = 0 #  Round Trip Time in seconds
AVG_DELAY = 0 # Average user delay
TRAFFIC = 0 # Edge-cloud traffic in Mbps
APP = np.array([]) # Microservice instance-set names
RCPU = np.array([]) # CPU provided to each instance-set
RMEM = np.array([]) # CPU provided to each instance-set
RCPU_EDGE = 0 # CPU provided to each instance-set in the edge cluster
RCPU_CLOUD = 0 # CPU provided to each instance-set in the cloud cluster
SLO_MARGIN_UNOFFLOAD = 0.8 # Service Level Objective increase margin
APP_EDGE = np.array([]) # Instance-set in the edge cluster

# Connect to Prometheus
prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

# Function that get microservice instance-set names of the application
def get_app_names():
    while True:
        global APP
        command1 = f'kubectl get deployment -n {NAMESPACE} -o custom-columns=APP:.metadata.labels.app --no-headers=true | grep -v "<none>" | sort -t \'s\' -k2,2n | uniq;'
        result1 = subprocess.run(command1, shell=True, check=True, text=True, stdout=subprocess.PIPE)
        output1 = result1.stdout
        # Split the output by newline, remove any leading/trailing whitespaces, and filter out empty strings
        APP = [value.strip() for value in output1.split('\n') if value.strip()]
        time.sleep(5)
        #return app_names

# Function that get the average delay from the istio-ingress gateway
def get_avg_delay():
    while True:
        global AVG_DELAY
        # Query to obtain the average delay
        query_avg_delay = f'sum by (source_workload) (rate(istio_request_duration_milliseconds_sum{{source_workload="istio-ingressgateway", reporter="source", response_code="200"}}[1m])) / sum by (source_workload) (rate(istio_request_duration_milliseconds_count{{source_workload="istio-ingressgateway", reporter="source", response_code="200"}}[1m]))'
        result_query = prom.custom_query(query=query_avg_delay)
        if result_query:
            for result in result_query:
                avg_delay = result["value"][1]  # extract avg_delay result
                # Check if the value is NaN
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

# Function that get the requests per second
def get_lamba():
    # Query to obtain requests per second
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{source_app="istio-ingressgateway", source_cluster="cluster2", reporter="destination", response_code="200"}}[{PERIOD}m]))'
    query_result = prom.custom_query(query=query_lambda)
    if query_result:
        for result in query_result:
            return float(result["value"][1]) # Get the value of lambda

# Function that get the CPU provided to each microservice instance-set
def get_Rcpu():
    time.sleep(5)
    global RCPU, RCPU_EDGE, RCPU_CLOUD
    while True:
        app_names = APP # Microservice instance-set names
        Rcpu_cloud = np.full(len(app_names), -1, dtype=float) # Initialize Rcpu_cloud
        Rcpu_edge = np.full(len(app_names), -1, dtype=float) # Initialize Rcpu_edge
        
        # Query to obtain cpu provided to each instance-set in the cloud cluster
        query_cpu_cloud = f'sum by (container) (last_over_time(kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="cpu", container!="istio-proxy", cluster!="cluster2"}}[1m]))'
        cpu_cloud_results = prom.custom_query(query=query_cpu_cloud)
        if cpu_cloud_results:
            for result in cpu_cloud_results:
                Rcpu_value = result["value"][1] # Get the value of Rcpu
                Rcpu_cloud[app_names.index(result["metric"]["container"])] = float(Rcpu_value) # Insert the value Rcpu_cloud in the correct position of the array
        
        # Query to obtain cpu provided to each instance-set in the edge cluster
        query_cpu_edge = f'sum by (container) (last_over_time(kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="cpu", container!="istio-proxy", cluster="cluster2"}}[1m]))'
        cpu_edge_results = prom.custom_query(query=query_cpu_edge)
        if cpu_edge_results:
            for result in cpu_edge_results:
                Rcpu_value = result["value"][1] # Get the value of Rcpu
                Rcpu_edge[app_names.index(result["metric"]["container"])] = float(Rcpu_value) # Insert the value Rcpu_edge in the correct position of the array
        
        Rcpu_edge_graphs = Rcpu_edge.copy() # Copy Rcpu_edge array to Rcpu_edge_graphs array
        
        # Check for missing values in Rcpu_edge and replace them with corresponding values from Rcpu_cloud
        for i, value in enumerate(Rcpu_edge):
            if value == -1:
                Rcpu_edge[i] = Rcpu_cloud[i]

        # Check for missing values in Rcpu_edge_graphs and replace them with 0 for graphs
        for i, value in enumerate(Rcpu_edge_graphs):
            if value == -1:
                Rcpu_edge_graphs[i] = 0

        # Save Rcpu values
        RCPU_EDGE = np.sum(Rcpu_edge_graphs)
        RCPU_CLOUD = np.sum(Rcpu_cloud)
        RCPU = np.concatenate((np.append(Rcpu_cloud, 0), np.append(Rcpu_edge, 0)))
        time.sleep(5)
    #return Rcpu

# Function that get the memory provided to each microservice instance-set
def get_Rmem():
    global RMEM
    while True:
        app_names = APP # Microservice instance-set names
        Rmem_cloud = np.full(len(app_names), -1, dtype=float) # Initialize Rmem_cloud
        Rmem_edge = np.full(len(app_names), -1, dtype=float) # Initialize Rmem_edge
        
        # Query to obtain memory provided to each instance-set in the cloud cluster
        query_mem_cloud = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="memory", container!="istio-proxy",cluster!="cluster2"}})'
        mem_cloud_results = prom.custom_query(query=query_mem_cloud)
        if mem_cloud_results:
            for result in mem_cloud_results:
                Rmem_value = result["value"][1] # Get the value of Rmem
                Rmem_cloud[app_names.index(result["metric"]["container"])] = float(Rmem_value) # Insert the value Rmem_cloud in the correct position of the array
        else:
            Rmem_cloud = np.zeros(len(app_names))

        # Query to obtain memory provided to each instance-set in the edge cluster
        query_mem_edge = f'sum by (container) (kube_pod_container_resource_limits{{namespace="{NAMESPACE}", resource="memory", container!="istio-proxy",cluster="cluster2"}})'
        mem_edge_results = prom.custom_query(query=query_mem_edge)
        if mem_edge_results:
            for result in mem_edge_results:
                Rmem_value = result["value"][1] # Get the value of Rmem
                Rmem_edge[app_names.index(result["metric"]["container"])] = float(Rmem_value) # Insert the value Rmem_edge in the correct position of the array
        else:
            Rmem_cloud = np.zeros(len(app_names))

        # Check for missing values in Rmem_edge and replace them with corresponding values from Rmem_cloud
        for i, value in enumerate(Rmem_edge):
            if value == -1:
                Rmem_edge[i] = Rmem_cloud[i]

        RMEM = np.concatenate((np.append(Rmem_cloud,0), np.append(Rmem_edge,0)))
        time.sleep(5)
    #return Rmem

# Function that get the instance-set already in edge cluster
def get_app_edge():
    global APP_EDGE
    time.sleep(5)
    while True:
        APP_EDGE = np.zeros(len(APP), dtype=int) # Initialize array with zeros with lenght equal to the number of instance-set
        command = f'kubectl get deployments.apps -n edge --context {CTX_CLUSTER2} -o custom-columns=APP:.metadata.labels.app --no-headers=true | grep -v "<none>" | sort | uniq' # Get instance-set offloaded
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
        output = result.stdout
        app_names_edge = [value.strip() for value in output.split('\n') if value.strip()]
        for microservice in app_names_edge:
            APP_EDGE[APP.index(microservice)] = 1 # Set value for microservice in edge to 1
        time.sleep(10)
    #return app_edge

# Function that get the response size of each microservice instance-set
def get_Rs():
    app_names = APP # Get app names with the relative function
    Rs = np.zeros(len(app_names), dtype=float) # inizialize Rs array
    # app_names combined with OR (|) for prometheus query
    combined_names = "|".join(app_names)
    # Query to obtain response size of each microservice    
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{response_code="200", destination_app=~"{combined_names}", reporter="source"}}[{PERIOD}m]))/sum by (destination_app) (increase(istio_response_bytes_count{{response_code="200", destination_app=~"{combined_names}", reporter="source"}}[{PERIOD}m]))'
    r1 = prom.custom_query(query=query_Rs)
    if r1:
        for result in r1:
            Rs_value = result["value"][1]
            if Rs_value == "NaN":
                Rs[app_names.index(result["metric"]["destination_app"])] = 0 #  If there isn't traffic
            else:
                Rs[app_names.index(result["metric"]["destination_app"])] = float(Rs_value) # Insert the value Rs in the correct position of the array
    return Rs

# Function that checks if some HPA is running for both clusters
def check_hpa():
    # Load the kube config file for the cloud cluster
    config.load_kube_config(context="{CTX_CLUSTER1}")
    
    # Create the API object for the cloud cluster
    api_cluster1 = client.AppsV1Api()
    autoscaling_api_cluster1 = client.AutoscalingV1Api()

    # Load the kube config file for the edge cluster
    config.load_kube_config(context="{CTX_CLUSTER2}")
    
    # Create the API object for the edge cluster
    api_cluster2 = client.AppsV1Api()
    autoscaling_api_cluster2 = client.AutoscalingV1Api()

    while True:
        global HPA_STATUS
        # Get the list of all deployments in the cloud cluster
        deployments_cluster1 = api_cluster1.list_namespaced_deployment(namespace=NAMESPACE)

        # Get the list of all deployments in the edge cluster
        deployments_cluster2 = api_cluster2.list_namespaced_deployment(namespace=NAMESPACE)

        all_hpa_satisfy_condition = True  # Assume initially that all HPAs satisfy the condition

        # Check the cloud cluster
        for deployment in deployments_cluster1.items:
            deployment_name = deployment.metadata.name

            # Get the list of associated HPAs for the deployment in the cloud cluster
            associated_hpas = autoscaling_api_cluster1.list_namespaced_horizontal_pod_autoscaler(namespace=NAMESPACE)

            for hpa in associated_hpas.items:
                hpa_name = hpa.metadata.name

                # Check if the HPA is associated with the current deployment
                if hpa.spec.scale_target_ref.name == deployment_name:
                    # Get the target CPU utilization percentage
                    cpu_utilization = hpa.spec.target_cpu_utilization_percentage

                    # Get the current status of the HPA
                    hpa_status = autoscaling_api_cluster1.read_namespaced_horizontal_pod_autoscaler_status(name=hpa_name, namespace=NAMESPACE)
                    current_cpu_utilization_percentage = hpa_status.status.current_cpu_utilization_percentage

                    if current_cpu_utilization_percentage is None:
                        current_cpu_utilization_percentage = 0

                    # Check if the current CPU usage is less than the target CPU for the HPA
                    if current_cpu_utilization_percentage >= cpu_utilization * HPA_MARGIN:
                        # If an HPA doesn't satisfy the condition, set the flag to False and break out of the loop
                        all_hpa_satisfy_condition = False
                        break

        # Check the edge cluster
        for deployment in deployments_cluster2.items:
            deployment_name = deployment.metadata.name

            # Get the list of associated HPAs for the deployment in the edge cluster
            associated_hpas = autoscaling_api_cluster2.list_namespaced_horizontal_pod_autoscaler(namespace=NAMESPACE)

            for hpa in associated_hpas.items:
                hpa_name = hpa.metadata.name

                # Check if the HPA is associated with the current deployment
                if hpa.spec.scale_target_ref.name == deployment_name:
                    # Get the target CPU utilization percentage
                    cpu_utilization = hpa.spec.target_cpu_utilization_percentage

                    # Get the current status of the HPA
                    hpa_status = autoscaling_api_cluster2.read_namespaced_horizontal_pod_autoscaler_status(name=hpa_name, namespace=NAMESPACE)
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
        time.sleep(15)

# Function that get the RTT edge-cloud
def get_RTT():
    global RTT
    while True:
        RTT_array = np.array([])

        # Repeat the RTT measurement 10 times
        for i in range(10):
            # Get pod name of the rtt-edge microservice
            command2 = f"kubectl get pods --context {CTX_CLUSTER2} -n edge | awk '/rtt-edge/ {{print $1}}' | head -n 1"
            result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)
            rtt_edge_pod_name = result2.stdout.strip()  # Clean the output

            # Get RTT edge-cloud
            RTT_command = f"kubectl exec -n edge --context {CTX_CLUSTER2} -it {rtt_edge_pod_name} -- bash -c 'curl --head -s -w %{{time_total}} http://rtt-cloud 2>/dev/null | tail -1'"
            RTT_result = subprocess.run(RTT_command, shell=True, capture_output=True, text=True)
            RTT = RTT_result.stdout.strip()  # Clean the output
            try:
                RTT = float(RTT)  
            except ValueError:
                continue
            # Add RTT value to the array
            RTT_array = np.append(RTT_array, RTT)
        threshold = 1.5
        # Remove outliers and save the mean value of RTT
        for i in range (10):
            std_dev = np.std(RTT_array)
            mean = np.mean(RTT_array)
            RTT_array = [x for x in RTT_array if abs(x - mean) < threshold * std_dev]
        rtt = np.mean(RTT_array)
        RTT = round(float(rtt), 4)
        time.sleep(5)

# Function that save the avg delay in csv file
def save_delay(delay_value):
    with open(f'/home/alex/Downloads/matlab/{FOLDER}/{PLACEMENT_TYPE}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([delay_value])

# Function that save CPU used in edge and cloud clusters in csv file
def save_Rcpu():
    global RCPU_CLOUD
    global RCPU_EDGE
    while True:
        with open(f'/home/alex/Downloads/matlab/{FOLDER}/cpu_cloud_{PLACEMENT_TYPE}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([RCPU_CLOUD])
        with open(f'/home/alex/Downloads/matlab/{FOLDER}/cpu_edge_{PLACEMENT_TYPE}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([RCPU_EDGE])
        time.sleep(1)

# Function that get the cloud-edge traffic and save it in csv file
def get_traffic():
    while True:
        global TRAFFIC
        # total requests that arrive to dst microservice from src microservice + requests that arrive to src microservice from dst microservice
        query = f'(sum((rate(istio_response_bytes_sum{{source_workload=~"s.*[0-12]-edge|istio-ingressgateway", destination_workload=~"s.*[0-12]-cloud", reporter="destination", response_code="200"}}[1m])))+sum((rate(istio_request_bytes_sum{{source_workload=~"s.*[0-12]-edge|istio-ingressgateway", destination_workload=~"s.*[0-12]-cloud", reporter="destination", response_code="200"}}[1m])))) *8 /1000000'
        r1 = prom.custom_query(query=query)
        # Extract values from querie
        if r1:
            for result in r1:
                TRAFFIC = round(float(result["value"][1]), 2)
                #print("traffic=",TRAFFIC,"Mbps")
                with open(f'/home/alex/Downloads/matlab/{FOLDER}/{PLACEMENT_TYPE}_traffic.csv', 'a', newline='') as file:
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

    # Start the thread that checks the RTT edge-cloud
    thread_RTT = threading.Thread(target=get_RTT) # Create thread
    thread_RTT.daemon = True # Daemonize thread
    thread_RTT.start()

    # Start the thread that get the app names of the application
    thread_RTT = threading.Thread(target=get_app_names) # Create thread
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

    # Start the thread that checks RMEM in clusters 
    thread_delay = threading.Thread(target=get_Rmem) # Create thread
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
        # Check if the HPA is not running and go ahead
        if HPA_STATUS == 0:
            print(f"\rCurrent avg_delay: {AVG_DELAY} ms")
            # Check if the avg_delay is greater than the SLO
            if AVG_DELAY > SLO:
                duration_counter = 0
                # Check if the avg_delay is greater than the SLO for a stabilization window
                while AVG_DELAY > SLO and duration_counter < stabilization_window_seconds and HPA_STATUS == 0:
                    time.sleep(1)
                    if AVG_DELAY > SLO:
                        duration_counter += 1
                    print(f"\r*Current avg_delay: {AVG_DELAY} ms")
                #  Offloading
                if duration_counter >= stabilization_window_seconds and HPA_STATUS == 0:
                    print(f"\rSLO not satisfied, offloading...")
                    Rs = get_Rs() # Response size of each microservice instance-set
                    lambda_value = get_lamba() # Average user requests per second
                    M = int(len(RCPU)/2) # Number of microservices
                    if PLACEMENT_TYPE == "OE_PAMP":
                        OE_PAMP_off(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE)
                    elif PLACEMENT_TYPE == "RANDOM":
                        random_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE)
                    elif PLACEMENT_TYPE == "MFU":
                        mfu_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE)
                    elif PLACEMENT_TYPE == "IA":
                        IA_placement(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE)
            # Check if the avg_delay is less than the SLO + margin
            elif AVG_DELAY <= SLO*SLO_MARGIN_UNOFFLOAD:
                duration_counter = 0
                # Check if the avg_delay is lower for a stabilization window
                while AVG_DELAY < SLO*SLO_MARGIN_UNOFFLOAD and duration_counter < stabilization_window_seconds and HPA_STATUS == 0:
                    time.sleep(1)
                    print(f"\r**Current avg_delay: {AVG_DELAY} ms")
                    duration_counter += 1
                if AVG_DELAY == 0:
                    continue
                # Unoffloading
                elif duration_counter >= stabilization_window_seconds and HPA_STATUS == 0 and APP_EDGE.any() != 0:
                    print(f"\rSLO not satisfied")
                    if PLACEMENT_TYPE == "OE_PAMP":
                        print(f"\rUnoffloading...")
                        Rs = get_Rs() # Response size of each microservice instance-set
                        lambda_value = get_lamba() # Average user requests per second
                        M = int(len(RCPU)/2) # Number of microservice instance-set
                        OE_PAMP_unoff(RTT, AVG_DELAY, APP, APP_EDGE, RCPU, RMEM, Rs, M, SLO, lambda_value, CTX_CLUSTER2, NAMESPACE, prom, SLO_MARGIN_UNOFFLOAD, PERIOD, MICROSERVICE_DIRECTORY, HPA_DIRECTORY, NE)  
            time.sleep(1)
        else:
            print(f"\rHPA running...")
            time.sleep(1)
if __name__ == "__main__":
    main()