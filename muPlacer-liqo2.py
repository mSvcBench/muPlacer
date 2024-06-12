import time
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import yaml
import numpy as np
import re
from statemachine import StateMachine, State


def update_Rcpu():
    #time.sleep(5)
    global mu_config, prom, metrics
    #while True:
    now = time.time()
    areas = ['cloud-area','edge-area']
    services=metrics['services']
    # clean the Rcpu values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Rcpu'][area]['value'] = 0
            service['Rcpu'][area]['last-update'] = now
    for area in areas:
        pod_regex = metrics['global-regex'][area]['pod']
        query_cpu = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{cluster="{mu_config['spec'][area]['cluster']}",namespace="{mu_config['spec']['namespace']}",pod=~"{pod_regex}"}}[{query_period_str}]))'
        query_results = prom.custom_query(query=query_cpu)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['Rcpu'][area]['value'] = service['Rcpu'][area]['value'] + float(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service

def update_Rmem():
    global mu_config, prom, metrics

    now = time.time()
    areas = ['cloud-area','edge-area']
    services=metrics['services']
    # clean the Rmem values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Rmem'][area]['value'] = 0
            service['Rmem'][area]['last-update'] = now
    
    for area in areas:
        pod_regex = metrics['global-regex'][area]['pod']
        query_mem = f'sum by (pod) (container_memory_usage_bytes{{cluster="{mu_config['spec'][area]['cluster']}", namespace="{mu_config['spec']['namespace']}",pod=~"{pod_regex}"}})'
        query_results = prom.custom_query(query=query_mem)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['Rmem'][area]['value'] = service['Rmem'][area]['value'] + float(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service

# Function that get the requests per second
def update_lambda():
    global mu_config, prom, metrics

    destination_app_regex = "|".join(metrics['services'].keys())
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{cluster="{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period_str}]))'
    query_result = prom.custom_query(query=query_lambda)
    lambda_data = metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']
    now = time.time()
    if query_result:
        for result in query_result:
            if result["value"][1]=="NaN":
                value = 0
            else:
                value = float(result["value"][1])
            if now > lambda_data['last-update']:
                lambda_data['value'] = value
                lambda_data['last-update'] = now
            else:
                lambda_data['value'] = lambda_data['value'] + value
    return

# Function that get the response size of each microservice instance-set
def update_Rs():
    global mu_config, prom, metrics

    now = time.time()
    # clean Rs values
    services=metrics['services']
    for service_name in services:
        service=services[service_name]
        service['Rs']['value'] = 0
        service['Rs']['last-update'] = now

    destination_app_regex = "|".join(metrics['services'].keys())
    cluster_regex = mu_config['spec']['cloud-area']['cluster']+"|"+mu_config['spec']['edge-area']['cluster'] 
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))/sum by (destination_app) (increase(istio_response_bytes_count{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))'
    r1 = prom.custom_query(query=query_Rs)

    if r1:
        for result in r1:
            service_name = result["metric"]["destination_app"]
            if service_name in metrics['services']:
                service=metrics['services'][service_name]
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])
                service['Rs']['value'] = service['Rs']['value'] + value
    return

# Function that build call frequency matrix Fcm

def update_Fcm():
    global mu_config, prom, metrics
    
    now = time.time()

    # clean lambda and Fcm values
    services=metrics['services']
    for service_name in services:
        service=services[service_name]
        service['lambda']['value'] = 0
        service['lambda']['last-update'] = now
        service['Fcm']['value'] = dict()
        service['Fcm']['last-update'] = now

    destination_app_regex = "|".join(metrics['services'].keys())
    cluster_regex = mu_config['spec']['cloud-area']['cluster']+"|"+mu_config['spec']['edge-area']['cluster']
    source_app_regex = mu_config['spec']['edge-area']['istio-ingress-source-app']+"|"+destination_app_regex
    fcm_query_num=f'sum by (source_app,destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    lambda_query=f'sum by (destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{mu_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    r = prom.custom_query(query=lambda_query)
    r2 = prom.custom_query(query=fcm_query_num)
    for result in r:
        destination_app = result["metric"]["destination_app"]
        if destination_app in metrics['services']:
            if result["value"][1]=="NaN":
                value = 0
            else:
                value = float(result["value"][1])
            metrics['services'][destination_app]['lambda']['value'] = value
            metrics['services'][destination_app]['lambda']['last-update'] = now
            continue
        if destination_app == mu_config['spec']['edge-area']['istio-ingress-source-app']:
            if result["value"][1]=="NaN":
                value = 0
            else:
                value = float(result["value"][1])
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value'] = value
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['last-update'] = now
    
    for result in r2:
        source_app = result["metric"]["source_app"]
        destination_app = result["metric"]["destination_app"]
        if source_app in metrics['services'] and destination_app in metrics['services']:
            if metrics['services'][source_app]['lambda']['value'] == 0:
                value = 0
            else:
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])/float(metrics['services'][source_app]['lambda']['value'])
            metrics['services'][source_app]['Fcm']['value'][destination_app] = value
            metrics['services'][source_app]['Fcm']['last-update'] = now
            continue
        if source_app == mu_config['spec']['edge-area']['istio-ingress-source-app'] and destination_app in metrics['services']:
            if metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value'] == 0:
                value = 0
            else:
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])/metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value']
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['Fcm']['value'][destination_app] = value
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['Fcm']['last-update'] = now
    return

# Function that get the average delay from the istio-ingress gateway
def update_delay():
    global mu_config, prom, metrics
    destination_app_regex = "|".join(metrics['services'].keys())
    query_avg_delay = f'sum by (source_app) (rate(istio_request_duration_milliseconds_sum{{cluster=~"{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period_str}])) / sum by (source_app) (rate(istio_request_duration_milliseconds_count{{cluster=~"{mu_config['spec']['edge-area']['cluster']}", namespace="{mu_config['spec']['namespace']}", source_app="{mu_config['spec']['edge-area']['istio-ingress-source-app']}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period_str}]))'
    result_query = prom.custom_query(query=query_avg_delay)
    now = time.time()
    if result_query:
        for result in result_query:
            if result["value"][1]=="NaN":
                value=0
            else:
                value=float(result["value"][1])
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['value'] = value  # extract avg_delay result
            metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['last-update'] = now

def update_replicas():
    global mu_config, prom, metrics
    now = time.time()
    areas = ['cloud-area','edge-area']
    services=metrics['services']
    
    # clean the replicas values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['Replicas'][area]['value'] = 0
            service['Replicas'][area]['last-update'] = now

    for area in areas:
        deployment_regex = metrics['global-regex'][area]['deployment']
        # Query to obtain cpu provided to each instance-set in the cloud cluster
        query_replicas = f'kube_deployment_status_replicas{{cluster="{mu_config['spec'][area]['cluster']}",namespace="{mu_config['spec']['namespace']}",deployment=~"{deployment_regex}"}}'
        query_results = prom.custom_query(query=query_replicas)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['deployment'], result['metric']['deployment'], re.IGNORECASE):
                        service['Replicas'][area]['value'] = service['Replicas'][area]['value'] + int(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service

def get_regex():
    global mu_config, metrics
    # compute the pod/deployment regex for each service
    for sc in mu_config['spec']['services']:
        # compute the pod regex for the edge area
        items = sc['instances']['cloud-yamls']
        s = metrics['services'][sc['name']]
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
                        if metrics['global-regex']['cloud-area']['pod'] == '':
                            metrics['global-regex']['cloud-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            metrics['global-regex']['cloud-area']['pod'] = f'{metrics['global-regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if metrics['global-regex']['cloud-area']['deployment'] == '':
                            metrics['global-regex']['cloud-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global-regex']['cloud-area']['deployment'] = f'{metrics['global-regex']['cloud-area']['deployment']}|{partial_yaml['metadata']['name']}'
                items = sc['instances']['cloud-yamls']
        
        items = sc['instances']['edge-yamls']
        s = metrics['services'][sc['name']]
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
                        if metrics['global-regex']['edge-area']['pod'] == '':
                            metrics['global-regex']['edge-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            metrics['global-regex']['edge-area']['pod'] = f'{metrics['global-regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        if metrics['global-regex']['edge-area']['deployment'] == '':
                            metrics['global-regex']['edge-area']['deployment'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global-regex']['edge-area']['deployment'] = f'{metrics['global-regex']['edge-area']['deployment']}|{partial_yaml['metadata']['name']}'

def time_to_ms_converter(delay_string):
    if delay_string.endswith("ms"):
        value = int(delay_string.split("ms")[0]) 
    elif delay_string.endswith("s"):
        value = int(delay_string.split("s")[0])*1000
    elif delay_string.endswith("m"):
        value = int(delay_string.split("m")[0])*60000
    elif delay_string.endswith("h"):    
        value = int(delay_string.split("h")[0])*3600000
    return value

def init():
    global mu_config, metrics
    metrics = dict() # Global Status dictionary
    metrics['services'] = dict() # Services status dictionary
    istio_ingress = mu_config['spec']['edge-area']['istio-ingress-source-app']
    metrics[istio_ingress]=dict() # Istio-ingress status dictionary
    metrics['global-regex'] = dict() # Global regex dictionary
    metrics['global-regex']['edge-area'] = dict()
    metrics['global-regex']['cloud-area'] = dict()
    metrics['global-regex']['edge-area']['pod'] = ''
    metrics['global-regex']['edge-area']['deployment'] = ''
    metrics['global-regex']['cloud-area']['pod'] = ''
    metrics['global-regex']['cloud-area']['deployment'] = ''
    metrics['n-services'] = 0 # number of microservices

    # Initialize the service status information
    mid = 0 # microservice id
    services=metrics['services']
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
            
            # Initialize HPA
            services[s['name']]['hpa'] = dict()
            services[s['name']]['hpa']['cpu-utilization'] = s['hpa']['cpu-utilization']
            services[s['name']]['hpa']['min-replicas'] = s['hpa']['min-replicas']
            services[s['name']]['hpa']['max-replicas'] = s['hpa']['max-replicas']
            services[s['name']]['hpa']['info'] = 'HPA cpu configuration'
            
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
            services[s['name']]['lambda']['info'] = 'Service request rate req/s'

            # Inintialize the regex values
            services[s['name']]['regex'] = dict()
            services[s['name']]['regex']['edge-area'] = dict()
            services[s['name']]['regex']['cloud-area'] = dict()
            services[s['name']]['regex']['edge-area']['pod'] = ''
            services[s['name']]['regex']['edge-area']['deployment'] = ''
            services[s['name']]['regex']['cloud-area']['pod'] = ''
            services[s['name']]['regex']['cloud-area']['deployment'] = ''


    # Initialize Istio-ingress status
    metrics[istio_ingress]['Fcm'] = dict() # call frequency matrix
    metrics[istio_ingress]['Fcm']['info'] = 'Call frequency matrix'
    metrics[istio_ingress]['Fcm']['value'] = dict() 
    metrics[istio_ingress]['Fcm']['last-update'] = 0 # last update time
    metrics[istio_ingress]['delay'] = dict() # average delay in milliseconds
    metrics[istio_ingress]['delay']['value'] = 0
    metrics[istio_ingress]['delay']['info'] = 'Average edge user delay in ms'
    metrics[istio_ingress]['delay']['last-update'] = 0 # last update time
    metrics[istio_ingress]['lambda'] = dict() # average delay in milliseconds
    metrics[istio_ingress]['lambda']['value'] = 0
    metrics[istio_ingress]['lambda']['info'] = 'Request rate from edge user in req/s'
    metrics[istio_ingress]['lambda']['last-update'] = 0 # last update time

    metrics['n-services'] = mid+1 # number of microservices
    # Initialize global call frequency matrix

    metrics['global-Fcm'] = dict()
    metrics['global-Fcm']['info'] = 'Call frequency matrix'
    metrics['global-Fcm']['value'] = np.zeros((metrics['n-services'],metrics['n-services']))
    metrics['global-Fcm']['last-update'] = 0 # last update time
    metrics['global-Rs'] = dict()
    metrics['global-Rs']['info'] = 'Response size vector in bytes'
    metrics['global-Rs']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Rs']['last-update'] = 0 # last update time
    metrics['global-Replicas'] = dict()
    metrics['global-Replicas']['cloud-area'] = dict()
    metrics['global-Replicas']['cloud-area']['info'] = 'Replicas vector for cloud area'
    metrics['global-Replicas']['cloud-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Replicas']['cloud-area']['last-update'] = 0 # last update time
    metrics['global-Replicas']['edge-area'] = dict()
    metrics['global-Replicas']['edge-area']['info'] = 'Replicas vector for edge area'
    metrics['global-Replicas']['edge-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Replicas']['edge-area']['last-update'] = 0 # last update time
    metrics['global-Rcpu'] = dict()
    metrics['global-Rcpu']['cloud-area'] = dict()
    metrics['global-Rcpu']['cloud-area']['info'] = 'Call frequency matrix'
    metrics['global-Rcpu']['cloud-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Rcpu']['cloud-area']['last-update'] = 0 # last update time
    metrics['global-Rcpu']['edge-area'] = dict()
    metrics['global-Rcpu']['edge-area']['info'] = 'Call frequency matrix'
    metrics['global-Rcpu']['edge-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Rcpu']['edge-area']['last-update'] = 0 # last update time
    metrics['global-Rmem'] = dict()
    metrics['global-Rmem']['cloud-area'] = dict()
    metrics['global-Rmem']['cloud-area']['info'] = 'Call frequency matrix'
    metrics['global-Rmem']['cloud-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Rmem']['cloud-area']['last-update'] = 0 # last update time
    metrics['global-Rmem']['edge-area'] = dict()
    metrics['global-Rmem']['edge-area']['info'] = 'Call frequency matrix'
    metrics['global-Rmem']['edge-area']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-Rmem']['edge-area']['last-update'] = 0 # last update time
    metrics['global-lambda'] = dict()   
    metrics['global-lambda']['info'] = 'Request rate vector in req/s'
    metrics['global-lambda']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-lambda']['last-update'] = 0 # last update time
    metrics['global-delay'] = dict()
    metrics['global-delay']['info'] = 'Average delay vector in ms'
    metrics['global-delay']['value'] = np.zeros((metrics['n-services'],1))
    metrics['global-delay']['last-update'] = 0 # last update time

    

    # Get the pod/deployment regex for each service
    get_regex()


def check_hpa():
    global mu_config, metrics
    services = metrics['services']
    for service_name in services:
        service = services[service_name]
        cond1 = service['Replicas']['cloud-area']['value']>0 and service['Rcpu']['cloud-area']['value'] > service['hpa']['cpu-utilization'] and service['Replicas']['cloud-area']['value'] < service['hpa']['max-replicas'] # HPA shoud increase replicas
        cond2 = service['Replicas']['cloud-area']['value']>0 and service['Rcpu']['cloud-area']['value'] < service['hpa']['cpu-utilization'] and service['Replicas']['cloud-area']['value'] > service['hpa']['min-replicas'] # HPA shoud decrease replicas
        cond3 = service['Replicas']['edge-area']['value']>0 and service['Rcpu']['edge-area']['value'] > service['hpa']['cpu-utilization'] and service['Replicas']['edge-area']['value'] < service['hpa']['max-replicas'] # HPA shoud increase replicas
        cond4 = service['Replicas']['edge-area']['value']>0 and service['Rcpu']['edge-area']['value'] < service['hpa']['cpu-utilization'] and service['Replicas']['edge-area']['value'] > service['hpa']['min-replicas'] # HPA shoud decrease replicas
        if (cond1 or cond2 or cond3 or cond4):
            return True
    return False

class AutoplacerStataMachine(StateMachine):
    hpa_running = State('hpa-running', value=1)   # HPA is runnning
    camping = State('camping', initial=True, value=2) # periodic monitoring of the system and no need to take action
    offload_alarm = State('offload-alarm', value=3)   # offload delay threshold is reached; check if this state persist for a while
    unoffload_alarm = State('unoffload-alarm', value=4) # unoffload delay threshold is reached; check if this state persist for a while
    offloading = State('offloading', value=5) # offload in progress
    unoffloading = State('unoffloading', value=6) # unoffload in progress

    # possible state transition
    cycle = (
        hpa_running.to(camping)
        | camping.to(hpa_running)
        | camping.to(offload_alarm)
        | offload_alarm.to(camping)
        | offload_alarm.to(offloading)
        | camping.to(unoffload_alarm)
        | unoffload_alarm.to(camping)
        | unoffload_alarm.to(unoffloading)
        | unoffloading.to(camping)
        | offloading.to(camping)
        | camping.to.itself()
    )


    def on_enter_camping(self):
        print('Camping')
        update_delay()
        if metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['value'] > offload_delay_threshold_ms:
            self.to_offload_alarm()
        elif metrics[mu_config['spec']['edge-area']['istio-ingress-source-app']]['delay']['value'] < unoffload_delay_threshold_ms:
            self.to_unoffload_alarm()
        else:
            time.sleep(sync_period_sec)
            self.to_camping()
        return


def main():
    global mu_config, prom, metrics, query_period_str, sync_period_sec, stabilizaiton_window_sec, offload_delay_threshold_ms, unoffload_delay_threshold_ms

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
    sync_period_sec = time_to_ms_converter(mu_config['spec']['sync-period'])*1000
    query_period_str = mu_config['spec']['query-period']
    stabilizaiton_window_sec = time_to_ms_converter(mu_config['spec']['stabilization-window'])*1000
    offload_delay_threshold_ms = time_to_ms_converter(mu_config['spec']['offload-delay-threshold'])
    unoffload_delay_threshold_ms = time_to_ms_converter(mu_config['spec']['unoffload-delay-threshold'])

    # Initialize the microservice status dictionary
    init()

    # update loops
    # main loop
    sm = AutoplacerStataMachine()
    while True:
        update_Rcpu()
        update_Rmem()
        update_Rs()
        update_lambda()
        update_Fcm()
        update_delay()
        update_replicas()
        x = check_hpa()
        time.sleep(120)

    print(mu_config)

    # Your code here
    # Use the data_dict as needed

if __name__ == "__main__":
    main()