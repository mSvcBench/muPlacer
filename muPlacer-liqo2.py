import time
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import yaml
import numpy as np
import re
from statemachine import StateMachine, State


def get_Rcpu():
    #time.sleep(5)
    global mu_config, prom, status
    #while True:
    now = time.time()
    areas = ['cloud-area','edge-area']
    services=status['services']
    # clean the Rcpu values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Rcpu'][area]['value'] = 0
            service['Rcpu'][area]['last-update'] = now
    for area in areas:
        pod_regex = status['global-regex'][area]['pod']
        # Query to obtain cpu provided to each instance-set in the cloud cluster
        query_cpu = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{cluster="{mu_config['spec'][area]['cluster']}",namespace="{mu_config['spec']['namespace']}",pod=~"{pod_regex}"}}[{query_period}]))'
        query_results = prom.custom_query(query=query_cpu)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['Rcpu'][area]['value'] = str(float(service['Rcpu'][area]['value']) + float(result["value"][1])) # Add the consumed CPU of the Pod to the consumed CPU of the service

def get_Rmem():
    global mu_config, prom, status

    now = time.time()
    areas = ['cloud-area','edge-area']
    services=status['services']
    # clean the Rmem values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Rmem'][area]['value'] = 0
            service['Rmem'][area]['last-update'] = now
    # Query to obtain cpu provided to each instance-set in the cloud cluster
    for area in areas:
        pod_regex = status['global-regex'][area]['pod']
        query_mem = f'sum by (pod) (container_memory_usage_bytes{{cluster="{mu_config['spec'][area]['cluster']}", namespace="{mu_config['spec']['namespace']}",pod=~"{pod_regex}"}})'
        query_results = prom.custom_query(query=query_mem)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['Rmem'][area]['value'] = str(float(service['Rmem'][area]['value']) + float(result["value"][1])) # Add the consumed CPU of the Pod to the consumed CPU of the service

# Function that get the requests per second
def get_lambda():
    global mu_config, prom, status

    destination_app_regex = "|".join(status['services'].keys())
    # Query to obtain requests per second
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{cluster="{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period}]))'
    query_result = prom.custom_query(query=query_lambda)
    lambda_data = status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']
    now = time.time()
    if query_result:
        for result in query_result:
            if now > lambda_data['last-update']:
                lambda_data['value'] = result["value"][1]
                lambda_data['last-update'] = now
            else:
                lambda_data['value'] = lambda_data['value'] + result["value"][1]
    return

# Function that get the response size of each microservice instance-set
def get_Rs():
    global mu_config, prom, status

    now = time.time()
    # clean Rs values
    services=status['services']
    for service_name in services:
        service=services[service_name]
        service['Rs']['value'] = 0
        service['Rs']['last-update'] = now

    # app_names combined with OR (|) for prometheus query
    destination_app_regex = "|".join(status['services'].keys())
    cluster_regex = mu_config['spec']['cloud-area']['cluster']+"|"+mu_config['spec']['edge-area']['cluster']
    # Query to obtain response size of each microservice    
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period}]))/sum by (destination_app) (increase(istio_response_bytes_count{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period}]))'
    r1 = prom.custom_query(query=query_Rs)

    if r1:
        for result in r1:
            service_name = result["metric"]["destination_app"]
            if service_name in status['services']:
                service=status['services'][service_name]
                service['Rs']['value'] = str(float(service['Rs']['value']) + float(result["value"][1]))
    return

# Function that build call frequency matrix Fcm

def get_Fcm():
    global mu_config, prom, status
    
    now = time.time()

    # clean lambda and Fcm values
    services=status['services']
    for service_name in services:
        service=services[service_name]
        service['lambda']['value'] = 0
        service['lambda']['last-update'] = now
        service['Fcm']['value'] = dict()
        service['Fcm']['last-update'] = now

    destination_app_regex = "|".join(status['services'].keys())
    cluster_regex = mu_config['spec']['cloud-area']['cluster']+"|"+mu_config['spec']['edge-area']['cluster']
    source_app_regex = mu_config['spec']['edge-area']['istio-ingress-source-app']+"|"+destination_app_regex
    fcm_query_num=f'sum by (source_app,destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period}])) '
    lambda_query=f'sum by (destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period}])) '
    r = prom.custom_query(query=lambda_query)
    r2 = prom.custom_query(query=fcm_query_num)
    for result in r:
        destination_app = result["metric"]["destination_app"]
        if destination_app in status['services']:
            status['services'][destination_app]['lambda']['value'] = result["value"][1]
            status['services'][destination_app]['lambda']['last-update'] = now
            continue
        if destination_app == mu_config['spec']['edge-area']['istio-ingress-source-app']:
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value'] = result["value"][1]
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['last-update'] = now
    
    for result in r2:
        source_app = result["metric"]["source_app"]
        destination_app = result["metric"]["destination_app"]
        if source_app in status['services'] and destination_app in status['services']:
            value = str(float(result["value"][1])/float(status['services'][source_app]['lambda']['value']))
            status['services'][source_app]['Fcm']['value'][destination_app] = value
            status['services'][source_app]['Fcm']['last-update'] = now
            continue
        if source_app == mu_config['spec']['edge-area']['istio-ingress-source-app'] and destination_app in status['services']:
            value = str(float(result["value"][1])/float(status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value']))
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['Fcm']['value'][destination_app] = value
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['Fcm']['last-update'] = now
    return

# Function that get the average delay from the istio-ingress gateway
def get_avg_delay():
    global mu_config, prom, status
    destination_app_regex = "|".join(status['services'].keys())
    query_avg_delay = f'sum by (source_app) (rate(istio_request_duration_milliseconds_sum{{cluster=~"{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period}])) / sum by (source_app) (rate(istio_request_duration_milliseconds_count{{cluster=~"{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period}]))'
    result_query = prom.custom_query(query=query_avg_delay)
    now = time.time()
    if result_query:
        for result in result_query:
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['value'] = result["value"][1]  # extract avg_delay result
            status[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['last-update'] = now

def get_replicas():
    global mu_config, prom, status
    now = time.time()
    areas = ['cloud-area','edge-area']
    services=status['services']
    
    # clean the replicas values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Replicas'][area]['value'] = 0
            service['Replicas'][area]['last-update'] = now

    for area in areas:
        deployment_regex = status['global-regex'][area]['deployment']
        # Query to obtain cpu provided to each instance-set in the cloud cluster
        query_replicas = f'kube_deployment_status_replicas{{cluster="{mu_config['spec'][area]['cluster']}",namespace="{mu_config['spec']['namespace']}",deployment=~"{deployment_regex}"}}'
        query_results = prom.custom_query(query=query_replicas)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['deployment'], result['metric']['deployment'], re.IGNORECASE):
                        service['Replicas'][area]['value'] = str(float(service['Replicas'][area]['value']) + float(result["value"][1])) # Add the consumed CPU of the Pod to the consumed CPU of the service

#     if replicas_cloud_results:
#         for result in replicas_cloud_results:
#             if match := re.search(mu_config['spec']['cloud-area']['service-regex'], result['metric']['pod'], re.IGNORECASE):
#                 service_name = match.group(1)
#                 if service_name in services:
#                     service=services[service_name]  
#                     if now >  service['Rcpu']['cloud-area']['last-update']:
#                         # reset Rcpu values
#                         service['Rcpu']['cloud-area']['value'] = result["value"][1] # Add the consumed CPU of the Pod to the consumed CPU of the service
#                         service['Rcpu']['cloud-area']['last-update'] = now
#                     else:
#                         service['Rcpu']['cloud-area']['value'] = service['Rcpu']['cloud-area']['value'] + result["value"][1] # Add the consumed CPU of the Pod to the consumed CPU of the service
    
#     query_cpu_edge = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{cluster="{mu_config['spec']['edge-area']['cluster']}"namespace="{mu_config['spec']['namespace']}",pod=~"{mu_config['spec']['edge-area']['pod-regex']}"}}[{query_period}]))'
#     cpu_edge_results = prom.custom_query(query=query_cpu_edge)
#     if cpu_edge_results:
#         for result in cpu_edge_results:
#             if match := re.search(mu_config['spec']['edge-area']['service-regex'], result['metric']['pod'], re.IGNORECASE):
#                 service_name = match.group(1)
#                 if service_name in services:
#                     service=services[service_name]  
#                     if now > service['Rcpu']['edge-area']['value']:
#                         # reset Rcpu values
#                         service['Rcpu']['edge-area']['value'] = result["value"][1] # Add the consumed CPU of the Pod to the consumed CPU of the service
#                         service['Rcpu']['edge-area']['last-update'] = now
#                     else:
#                         service['Rcpu']['edge-area']['value'] = service['Rcpu']['edge-area']['value'] + result["value"][1] # Add the consumed CPU of the Pod to the consumed CPU of the service
            

# class AutoplacerStataMachine(StateMachine):
#     hpa = State('hpa', initial=True, value=1)
#     info = State('info', value=2)
#     warnining_offload = State('warning-offload', value=3)
#     warnining_unoffload = State('warning-offload', value=4)
#     offload = State('offload', value=5)
#     unoffload = State('unoffload', value=6)

def get_regex():
    global mu_config, status
    # compute the pod/deployment regex for each service
    for sc in mu_config['spec']['services']:
        # compute the pod regex for the edge area
        items = sc['instances']['cloud-yamls']
        s = status['services'][sc['name']]
        for item in items:
            yaml_to_apply = item
            with open(yaml_to_apply) as f:
                complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
                for partial_yaml in complete_yaml:
                    if partial_yaml['kind'] == 'Deployment' or partial_yaml['kind'] == 'StatefulSet':
                        if s['regex']['cloud-area']['pod'] == '':
                            s['regex']['cloud-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            s['regex']['cloud-area']['pod'] = f'{s['regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if s['regex']['cloud-area']['deployment'] == '':
                            s['regex']['cloud-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            s['regex']['cloud-area']['deployment'] = f'{s['regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}'
                        if status['global-regex']['cloud-area']['pod'] == '':
                            status['global-regex']['cloud-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            status['global-regex']['cloud-area']['pod'] = f'{status['global-regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if status['global-regex']['cloud-area']['deployment'] == '':
                            status['global-regex']['cloud-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            status['global-regex']['cloud-area']['deployment'] = f'{status['global-regex']['cloud-area']['deployment']}|{partial_yaml['metadata']['name']}'
                items = sc['instances']['cloud-yamls']
        
        items = sc['instances']['edge-yamls']
        s = status['services'][sc['name']]
        for item in items:
            yaml_to_apply = item
            with open(yaml_to_apply) as f:
                complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
                for partial_yaml in complete_yaml:
                    if partial_yaml['kind'] == 'Deployment' or partial_yaml['kind'] == 'StatefulSet':
                        if s['regex']['edge-area']['pod'] == '':
                            s['regex']['edge-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            s['regex']['edge-area']['pod'] = f'{s['regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if s['regex']['edge-area']['deployment'] == '':
                            s['regex']['edge-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            s['regex']['edge-area']['deployment'] = f'{s['regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}'
                        if status['global-regex']['edge-area']['pod'] == '':
                            status['global-regex']['edge-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            status['global-regex']['edge-area']['pod'] = f'{status['global-regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if status['global-regex']['edge-area']['deployment'] == '':
                            status['global-regex']['edge-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            status['global-regex']['edge-area']['deployment'] = f'{status['global-regex']['edge-area']['deployment']}|{partial_yaml['metadata']['name']}'
def status_init():
    global mu_config, status
    status = dict() # Global Status dictionary
    status['services'] = dict() # Services status dictionary
    istio_ingress = mu_config['spec']['edge-area']['istio-ingress-source-app']
    status[istio_ingress]=dict() # Istio-ingress status dictionary
    status['global-regex'] = dict() # Global regex dictionary
    status['global-regex']['edge-area'] = dict()
    status['global-regex']['cloud-area'] = dict()
    status['global-regex']['edge-area']['pod'] = ''
    status['global-regex']['edge-area']['deployment'] = ''
    status['global-regex']['cloud-area']['pod'] = ''
    status['global-regex']['cloud-area']['deployment'] = ''

    # Initialize the service status information
    mid = 0 # microservice id
    services=status['services']
    for s in mu_config['spec']['services']:
            services[s['name']]=dict()
            # Initialize the service id
            if mu_config['spec']['explicit-service-id']:
                services[s['name']]['id'] = s['id']
                if s['id'] > mid:
                    mid = s['id']+1 # needed for istio-ingress id
            else:
                services[s['name']]['id'] = mid
                mid = mid +1
            
            # Initialize the Rcpu values (actual cpu consumption)
            services[s['name']]['Rcpu'] = dict()
            services[s['name']]['Rcpu']['cloud-area'] = dict()
            services[s['name']]['Rcpu']['cloud-area']['value'] = 0
            services[s['name']]['Rcpu']['cloud-area']['last-update'] = 0
            services[s['name']]['Rcpu']['cloud-area']['info'] = 'CPU consumption in the cloud area in seconds per second'
            services[s['name']]['Rcpu']['edge-area'] = dict()
            services[s['name']]['Rcpu']['edge-area']['value'] = 0
            services[s['name']]['Rcpu']['edge-area']['last-update'] = 0
            services[s['name']]['Rcpu']['edge-area']['info'] = 'CPU consumption in the edge area in seconds per second'

            # Initialize the Rmem values (actuam memory consumption)
            services[s['name']]['Rmem'] = dict()
            services[s['name']]['Rmem']['cloud-area'] = dict()
            services[s['name']]['Rmem']['cloud-area']['info'] = 'Memory consumption in the cloud area in bytes'
            services[s['name']]['Rmem']['cloud-area']['value'] = 0
            services[s['name']]['Rmem']['cloud-area']['last-update'] = 0
            services[s['name']]['Rmem']['edge-area'] = dict()
            services[s['name']]['Rmem']['edge-area']['info'] = 'Memory consumption in the edge area in bytes'
            services[s['name']]['Rmem']['edge-area']['value'] = 0
            services[s['name']]['Rmem']['edge-area']['last-update'] = 0

            services[s['name']]['Replicas'] = dict()
            services[s['name']]['Replicas']['cloud-area'] = dict()
            services[s['name']]['Replicas']['cloud-area']['value'] = 0
            services[s['name']]['Replicas']['cloud-area']['last-update'] = 0
            services[s['name']]['Replicas']['cloud-area']['info'] = 'N. replicas in the cloud area'
            services[s['name']]['Replicas']['edge-area'] = dict()
            services[s['name']]['Replicas']['edge-area']['value'] = 0
            services[s['name']]['Replicas']['edge-area']['last-update'] = 0
            services[s['name']]['Replicas']['edge-area']['info'] = 'N. replicas in the edge area'

            # Initialize the Rs values (response size)
            services[s['name']]['Rs'] = dict()
            services[s['name']]['Rs']['value'] = 0
            services[s['name']]['Rs']['last-update'] = time.time()
            services[s['name']]['Rs']['info'] = 'Response size in bytes'

            # Initialize the Fcm values (calling frequency matrix)
            services[s['name']]['Fcm'] = dict()
            services[s['name']]['Fcm']['value'] = dict()
            services[s['name']]['Fcm']['last-update'] = 0
            services[s['name']]['Fcm']['info'] = 'Call frequency matrix'

            # Initialize the request rate values
            services[s['name']]['lambda'] = dict()
            services[s['name']]['lambda']['value'] = 0
            services[s['name']]['lambda']['last-update'] = 0
            services[s['name']]['lambda']['info'] = 'service request rate req/s'

            # Inintialize the regex values
            services[s['name']]['regex'] = dict()
            services[s['name']]['regex']['edge-area'] = dict()
            services[s['name']]['regex']['cloud-area'] = dict()
            services[s['name']]['regex']['edge-area']['pod'] = ''
            services[s['name']]['regex']['edge-area']['deployment'] = ''
            services[s['name']]['regex']['cloud-area']['pod'] = ''
            services[s['name']]['regex']['cloud-area']['deployment'] = ''


    # Initialize Istio-ingress status
    status[istio_ingress]['Fcm'] = dict() # call frequency matrix
    status[istio_ingress]['Fcm']['info'] = 'Call frequency matrix'
    status[istio_ingress]['Fcm']['value'] = dict() 
    status[istio_ingress]['Fcm']['last-update'] = 0 # last update time
    status[istio_ingress]['delay'] = dict() # average delay in milliseconds
    status[istio_ingress]['delay']['value'] = 0
    status[istio_ingress]['delay']['info'] = 'Average edge user delay in ms'
    status[istio_ingress]['delay']['last-update'] = 0 # last update time
    status[istio_ingress]['lambda'] = dict() # average delay in milliseconds
    status[istio_ingress]['lambda']['value'] = 0
    status[istio_ingress]['lambda']['info'] = 'Request rate from edge user in req/s'
    status[istio_ingress]['lambda']['last-update'] = 0 # last update time

    # Get the pod/deployment regex for each service
    get_regex()


def main():
    global mu_config, prom, status, query_period

    # Load the configuration YAML file
    with open('muPlacerConfig.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    # Convert the YAML data to a dictionary
    mu_config = dict(yaml_data)
    
    # Load Kubernetes configuration
    config.load_kube_config()

    # Create a Kubernetes API client
    k8s_apps_api = client.AppsV1Api()
    k8s_core_api = client.CoreV1Api()

    # Create a Prometheus client
    prom = PrometheusConnect(url=mu_config['spec']['prometheus-url'], disable_ssl=True)




    # Get the sync period, query period, and stabilization window
    sync_period = mu_config['spec']['sync-period']
    query_period = mu_config['spec']['query-period']
    stabilizaiton_window = mu_config['spec']['stabilization-window']

    # Initialize the microservice status dictionary
    status_init()

    # update loops
    # main loop
    while True:
        get_Rcpu()
        get_Rmem()
        get_Rs()
        get_lambda()
        get_Fcm()
        get_avg_delay()
        get_replicas()
        time.sleep(120)

    print(mu_config)

    # Your code here
    # Use the data_dict as needed

if __name__ == "__main__":
    main()