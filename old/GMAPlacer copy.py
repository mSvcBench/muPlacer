import argparse
import logging
import re
import sys
import time
import subprocess

import kubernetes
import numpy as np
import yaml

from os import environ
from prometheus_api_client import PrometheusConnect,PrometheusApiClientException

import old.EPAMP_offload_caching as EPAMP_offload_caching
import old.EPAMP_unoffload_from_void as EPAMP_unoffload_from_void
import SBMP_GMA_Connector

def update_acpu():
    global gma_config, prom_client, metrics

    logger.info(f"Update actual cpu utilization")

    now = time.time()

    areas = ['cloud-area','edge-area']
    services=metrics['services']
    # clean the acpu values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['acpu'][area]['value'] = 0
            service['acpu'][area]['last-update'] = now
    metrics['global']['acpu']['cloud-area']['value'] = np.zeros(M)
    
    # update values
    for area in areas:
        pod_regex = metrics['global']['regex'][area]['pod']
        query_cpu = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{cluster="{gma_config['spec'][area]['cluster']}",namespace="{gma_config['spec']['namespace']}",pod=~"{pod_regex}"}}[{query_period_str}]))'
        query_results = prom_client.custom_query(query=query_cpu)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['acpu'][area]['value'] = service['acpu'][area]['value'] + float(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service
                        service['acpu'][area]['last-update'] = now
                        metrics['global']['acpu'][area]['value'][service['id']] = service['acpu'][area]['value']
                        metrics['global']['acpu'][area]['last-update'] = now

def update_amem():
    global gma_config, prom_client, metrics

    logger.info(f"Update actual memory utilization")

    now = time.time()
    
    areas = ['cloud-area','edge-area']
    services=metrics['services']
    # clean the amem values
    for area in areas:
        for service_name in services:
            service=services[service_name]
            service['amem'][area]['value'] = 0
            service['amem'][area]['last-update'] = now
    metrics['global']['amem']['cloud-area']['value'] = np.zeros(M)
    
    # update values
    for area in areas:
        pod_regex = metrics['global']['regex'][area]['pod']
        query_mem = f'sum by (pod) (container_memory_usage_bytes{{cluster="{gma_config['spec'][area]['cluster']}", namespace="{gma_config['spec']['namespace']}",pod=~"{pod_regex}"}})'
        query_results = prom_client.custom_query(query=query_mem)
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        service['amem'][area]['value'] = service['amem'][area]['value'] + float(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service
                        service['amem'][area]['last-update'] = now
                        metrics['global']['amem'][area]['value'][service['id']] = service['amem'][area]['value']
                        metrics['global']['amem'][area]['last-update'] = now

def update_lambda():
    global gma_config, prom_client, metrics, M

    logger.info(f"Update request rate values")

    now = time.time()

    #clean lambda values
    metrics['global']['lambda']['value'][M-1] = 0  # M-1 is the ingex of the istio-ingress in the global lambda vector
    metrics[gma_config['spec']['edge-area']['istio-ingress-source-app']]['lambda']['value'] = 0
    
    # update lambda values
    destination_app_regex = "|".join(metrics['services'].keys())
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{cluster="{gma_config['spec']['edge-area']['cluster']}", namespace="{gma_config['spec']['namespace']}", source_app="{edge-istio-ingress-app}", destination_app=~"{destination_app_regex}", reporter="destination", response_code="200"}}[{query_period_str}]))'
    query_result = prom_client.custom_query(query=query_lambda)
    lambda_data = metrics[edge-istio-ingress-app]['lambda']
    
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
            
            metrics['global']['lambda']['value'][M-1] = lambda_data['value']  # M-1 is the index of the istio-ingress in the global lambda vector
            metrics['global']['lambda']['last-update'] = now
    return


def update_Rs():
    global gma_config, prom_client, metrics

    logger.info(f"Update response size values")

    now = time.time()

    # clean Rs values
    services=metrics['services']
    for service_name in services:
        service=services[service_name]
        service['rs']['value'] = 0
        service['rs']['last-update'] = now
    metrics['global']['rs']['value'] = np.zeros(M)

    # update Rs values
    destination_app_regex = "|".join(metrics['services'].keys())
    cluster_regex = gma_config['spec']['cloud-area']['cluster']+"|"+gma_config['spec']['edge-area']['cluster'] 
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{cluster=~"{cluster_regex}",namespace="{gma_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))/sum by (destination_app) (increase(istio_response_bytes_count{{cluster=~"{cluster_regex}",namespace="{gma_config['spec']['namespace']}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))'
    r1 = prom_client.custom_query(query=query_Rs)

    if r1:
        for result in r1:
            service_name = result["metric"]["destination_app"]
            if service_name in metrics['services']:
                service=metrics['services'][service_name]
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])
                service['rs']['value'] = service['rs']['value'] + value
            metrics['global']['rs']['value'][service['id']] = service['rs']['value']
            metrics['global']['rs']['last-update'] = now
    return

def update_Fcm():
    global gma_config, prom_client, metrics
    
    logger.info(f"Update call frequency matrix")

    now = time.time()

    # clean lambda and Fcm values
    services=metrics['services']
    for service_name in services:
        service=services[service_name]
        service['lambda']['value'] = 0
        service['lambda']['last-update'] = now
        service['fcm']['value'] = dict()
        service['fcm']['last-update'] = now
    metrics['global']['fcm']['value'] = np.zeros((M,M))
    metrics['global']['lambda']['value'][:-1] = 0 # M-1 is the index of the istio-ingress in the global lambda vector and this function does not update the lambda of the istio-ingress due to the reporter="destination" filter

    # update lambda and Fcm values
    destination_app_regex = "|".join(metrics['services'].keys())
    cluster_regex = gma_config['spec']['cloud-area']['cluster']+"|"+gma_config['spec']['edge-area']['cluster']
    source_app_regex = edge-istio-ingress-app+"|"+destination_app_regex
    fcm_query_num=f'sum by (source_app,destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{gma_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    lambda_query=f'sum by (destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{gma_config['spec']['namespace']}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    r = prom_client.custom_query(query=lambda_query)
    r2 = prom_client.custom_query(query=fcm_query_num)
    for result in r:
        destination_app = result["metric"]["destination_app"]
        if destination_app in metrics['services']:
            if result["value"][1]=="NaN":
                value = 0
            else:
                value = float(result["value"][1])
            metrics['services'][destination_app]['lambda']['value'] = value
            metrics['services'][destination_app]['lambda']['last-update'] = now
            metrics['global']['lambda']['value'][metrics['services'][destination_app]['id']] = value
            metrics['global']['lambda']['last-update'] = now
            continue

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
            metrics['services'][source_app]['fcm']['value'][destination_app] = value
            metrics['services'][source_app]['fcm']['last-update'] = now
            metrics['global']['fcm']['value'][metrics['services'][source_app]['id']][metrics['services'][destination_app]['id']] = value
            metrics['global']['fcm']['last-update'] = now
            continue
        if source_app == edge-istio-ingress-app and destination_app in metrics['services']:
            if metrics[edge-istio-ingress-app]['lambda']['value'] == 0:
                value = 0
            else:
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])/metrics[edge-istio-ingress-app]['lambda']['value']
            metrics[edge-istio-ingress-app]['fcm']['value'][destination_app] = value
            metrics[edge-istio-ingress-app]['fcm']['last-update'] = now
            metrics['global']['fcm']['value'][M-1][metrics['services'][destination_app]['id']] = value
            metrics['global']['fcm']['last-update'] = now
    return

# Function that get the average delay from the istio-ingress gateway
def update_delay():
    global gma_config, prom_client, metrics

    logger.info(f"Update delay values from istio ingress in the edge area")

    now = time.time()
    # clean the delay values
    metrics[edge-istio-ingress-app]['delay']['value'] = 0
    metrics['global']['delay']['edge-area']['value'][M-1] = 0

    # update the delay values
    destination_app_regex = "|".join(metrics['services'].keys())
    query_avg_delay = f'sum by (source_app) (rate(istio_request_duration_milliseconds_sum{{cluster=~"{gma_config['spec']['edge-area']['cluster']}", namespace="{gma_config['spec']['edge-area']['istio-ingress-namespace']}", source_app="{edge-istio-ingress-app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}])) / sum by (source_app) (rate(istio_request_duration_milliseconds_count{{cluster=~"{gma_config['spec']['edge-area']['cluster']}", namespace="{gma_config['spec']['edge-area']['istio-ingress-namespace']}", source_app="{edge-istio-ingress-app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}]))'
    result_query = prom_client.custom_query(query=query_avg_delay)
    
    if result_query:
        for result in result_query:
            if result["value"][1]=="NaN":
                value=0
            else:
                value=float(result["value"][1])
            metrics[edge-istio-ingress-app]['delay']['value'] = value  # extract avg_delay result
            metrics[edge-istio-ingress-app]['delay']['last-update'] = now
            metrics['global']['delay']['edge-area']['value'][M-1] = value
            metrics['global']['delay']['edge-area']['last-update'] = now

# def update_replicas():
#     global gpa_config, prom_client, metrics

#     logger.info(f"Update replicas values")
    
#     now = time.time()
    
#     areas = ['cloud-area','edge-area']
#     services=metrics['services']
#     # clean the replicas values
#     for area in areas:
#         for service_name in services:
#             service=services[service_name]
#             service['replicas'][area]['value'] = 0
#             service['replicas'][area]['last-update'] = now
#     metrics['global']['replicas']['cloud-area']['value'] = np.zeros(M)

#     # update values
#     for area in areas:
#         deployment_regex = metrics['global']['regex'][area]['deployment']
#         # Query to obtain cpu provided to each instance-set in the cloud cluster
#         query_replicas = f'last_over_time(kube_deployment_status_replicas{{cluster="{gpa_config['spec'][area]['cluster']}",namespace="{gpa_config['spec']['namespace']}",deployment=~"{deployment_regex}"}}[{query_period_str}])'
#         query_results = prom_client.custom_query(query=query_replicas)
#         if query_results:
#             for result in query_results:
#                 for service_name in services:
#                     service=services[service_name]
#                     if re.search(service['regex'][area]['deployment'], result['metric']['deployment'], re.IGNORECASE):
#                         service['replicas'][area]['value'] = service['replicas'][area]['value'] + int(result["value"][1]) # Add the consumed CPU of the Pod to the consumed CPU of the service
#                         service['replicas'][area]['last-update'] = now
#                         metrics['global']['replicas'][area]['value'][service['id']] = service['replicas'][area]['value']
#                         metrics['global']['replicas'][area]['last-update'] = now

def update_and_check_HPA():
    global gma_config, prom_client, metrics, k8s_apiclient

    logger.info(f"Update HPA values")

    now = time.time()

    namespace = gma_config['spec']['namespace'] 
    areas = ['cloud-area','edge-area']
    services=metrics['services']
    hpa_running = False
    
    # reset the hpa values
    for area in areas:    
        for service_name in services:
            service=services[service_name]
            service['hpa'][area]['current-replicas'] = 0
            service['hpa'][area]['desired-replicas'] = 0
            service['hpa'][area]['last-update'] = now
            metrics['global']['replicas'][area]['value'][service['id']] = 0
            metrics['global']['replicas'][area]['last-update'] = now
    
    # update the hpa values
    for area in areas:
        with k8s_apiclient[area] as api_client:
        # Create an instance of the API class
            api_instance = kubernetes.client.AutoscalingV1Api(api_client)
            try:
                api_response = api_instance.list_namespaced_horizontal_pod_autoscaler(namespace,pretty='True',)
            except kubernetes.client.rest.ApiException as e:
                print("Exception when calling AutoscalingV1Api->list_namespaced_horizontal_pod_autoscaler: %s\n" % e)
                return
            for hpa in api_response.items:
                if re.search(metrics['global']['regex'][area]['hpa'], hpa.metadata.name, re.IGNORECASE):
                    for service_name in metrics['services']:
                        service=metrics['services'][service_name]
                        if re.search(service['regex'][area]['hpa'], hpa.metadata.name, re.IGNORECASE):
                            service['hpa'][area]['min-replicas'] = hpa.spec.min_replicas
                            service['hpa'][area]['max-replicas'] = hpa.spec.max_replicas
                            service['hpa'][area]['desired-replicas'] = hpa.status.desired_replicas
                            service['hpa'][area]['current_replicas'] = hpa.status.current_replicas
                            service['hpa'][area]['last_update'] = now
                            metrics['global']['replicas'][area]['value'][service['id']] = hpa.status.current_replicas
                            metrics['global']['replicas'][area]['last-update'] = now
                            if hpa.status.desired_replicas!=hpa.status.current_replicas:
                                hpa_running = hpa_running or True
                            break
    return hpa_running

def update_metrics():
    update_acpu()
    update_amem()
    update_lambda()
    update_Rs()
    update_Fcm()
    update_delay()
    #update_replicas()
    return

def apply_configuration(result_list):
    global gma_config, metrics, service_id_to_name
    result_cloud_area = result_list[0] # result_list[0] contains cloud-area information
    result_edge_area = result_list[1] # result_list[1] contains edge-area information
    
    
    # remove resources from edge area
    for service_id in result_edge_area['to-delete']:
        if service_id not in service_id_to_name:
            continue
        name = service_id_to_name[service_id]
        service=metrics['services'][name]
        
        # move back replicas to cloud area
        workload_name = service['regex']['cloud-area']['workload']['regex']
        workload_type = service['regex']['cloud-area']['workload']['type']
        if workload_type != 'daemonset':
            cloud_replicas = service['hpa']['cloud-area']['current_replicas']+service['hpa']['edge-area']['current_replicas']
            if cloud_replicas > service['hpa']['cloud-area']['max-replicas']:
                cloud_replicas = service['hpa']['cloud-area']['max-replicas']
            command = f'kubectl --context {gma_config['spec']['cloud-area']['context']} -n {gma_config['spec']['namespace']} scale {workload_type} {workload_name} --replicas {cloud_replicas}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Scale {workload_type} {workload_name} in cloud-area to {cloud_replicas} replicas: {output}")            
        
        # delete resources in edge area
        for files in service['instances']['edge-yamls']:
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {gma_config['spec']['namespace']} delete -f {files}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Delete resource for service {name} in edge-area: {output}")

    # apply resources in edge area
    for service_id in result_edge_area['to-apply']:
        if service_id not in service_id_to_name:
            continue
        name = service_id_to_name[service_id]
        service=metrics['services'][name]
        for files in service['instances']['edge-yamls']:
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {gma_config['spec']['namespace']} apply -f {files}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Apply resource in edge-area: {output}")
        
        # clone replicas from cloud to edge area
        workload_name = service['regex']['edge-area']['workload']['regex']
        workload_type = service['regex']['edge-area']['workload']['type']
        if workload_type != 'daemonset':
            edge_replicas = gma_config['spec']['edge-area']['resource-scaling'] * service['hpa']['cloud-area']['current_replicas']
            if edge_replicas > service['hpa']['edge-area']['max-replicas']:
                edge_replicas = service['hpa']['edge-area']['max-replicas']
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {gma_config['spec']['namespace']} scale {workload_type} {workload_name} --replicas {edge_replicas}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Scale deployment {service['regex']['edge-area']['workload']['regex']} in edge-area to {edge_replicas} replicas: {output}")

    
    # for service_id in result_cloud_area['to-delete']:
    #     if service_id not in service_id_to_name:
    #         continue
    #     name = service_id_to_name[service_id]
    #     service=metrics['services'][name]
    #     for files in service['instances']['cloud-yamls']:
    #         command = f'kubectl --context {gpa_config['spec']['cloud-area']['context']} -n {gpa_config['spec']['namespace']} delete -f {files}'
    #         result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
    #         output = result.stdout
    #         logger.info(f"Delete resource in cloud-area: {output}")
    

    
    # for service_id in result_cloud_area['to-apply']:
    #     if service_id not in service_id_to_name:
    #         continue
    #     name = service_id_to_name[service_id]
    #     service=metrics['services'][name]
    #     for files in service['instances']['cloud-yamls']:
    #         command = f'kubectl --context {gpa_config['spec']['cloud-area']['context']} -n {gpa_config['spec']['namespace']} apply -f {files}'
    #         result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
    #         output = result.stdout
    #         logger.info(f"Apply resource in cloud-area: {output}")
        

    

    

    




def parse_yaml():
    global gma_config, metrics

    logger.info(f"Parse yaml files")

    # compute the pod/deployment regex for each service
    for sc in gma_config['spec']['services']:
        # compute the pod regex for the edge area
        items = sc['instances']['cloud-yamls']
        s = metrics['services'][sc['name']]
        for item in items:
            yaml_to_apply = item
            with open(yaml_to_apply) as f:
                complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
                for partial_yaml in complete_yaml:
                    if partial_yaml['kind'] == 'Deployment' or partial_yaml['kind'] == 'StatefulSet' or partial_yaml['kind'] == 'DaemonSet':
                        # update pod information
                        if s['regex']['cloud-area']['pod'] == '':
                            s['regex']['cloud-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            s['regex']['cloud-area']['pod'] = f'{s['regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        
                        if metrics['global']['regex']['cloud-area']['pod'] == '':
                            metrics['global']['regex']['cloud-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            metrics['global']['regex']['cloud-area']['pod'] = f'{metrics['global']['regex']['cloud-area']['pod']}|{partial_yaml['metadata']['name']}-.*'    
                        
                        # update workload information
                        if s['regex']['cloud-area']['workload']['regex'] != '':
                            logger.error(f"Multiple deployments/statefulset/daemonset for the service {sc['name']} in the cloud-area not supported")
                            exit(1)
                        if partial_yaml['kind'] == 'Deployment':
                            s['regex']['cloud-area']['workload']['type'] == 'deployment'
                            s['regex']['cloud-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                        if partial_yaml['kind'] == 'StatefulSet':
                            s['regex']['cloud-area']['workload']['type'] == 'statefulset'
                            s['regex']['cloud-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                        if partial_yaml['kind'] == 'DaemonSet':
                            s['regex']['cloud-area']['workload']['type'] == 'daemonset'
                            s['regex']['cloud-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                                                                                                                                  
                        if metrics['global']['regex']['cloud-area']['workload'] == '':
                            metrics['global']['regex']['cloud-area']['workload'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global']['regex']['cloud-area']['workload'] = f'{metrics['global']['regex']['cloud-area']['workload']}|{partial_yaml['metadata']['name']}'
                    
                    # update hpa information
                    if partial_yaml['kind'] == 'HorizontalPodAutoscaler' :
                        if s['regex']['cloud-area']['hpa'] == '':
                            s['regex']['cloud-area']['hpa'] = f'{partial_yaml['metadata']['name']}'
                            s['hpa']['cloud-area']['min-replicas'] = partial_yaml['spec']['minReplicas']
                            s['hpa']['cloud-area']['max-replicas'] = partial_yaml['spec']['maxReplicas']
                        else:
                            logger.error(f"Multiple HPA for the service {sc['name']} in the cloud-area not supported")
                            exit(1)
                        if metrics['global']['regex']['cloud-area']['hpa'] == '':
                            metrics['global']['regex']['cloud-area']['hpa'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global']['regex']['cloud-area']['hpa'] = f'{metrics['global']['regex']['cloud-area']['hpa']}|{partial_yaml['metadata']['name']}'
        
        items = sc['instances']['edge-yamls']
        s = metrics['services'][sc['name']]
        for item in items:
            yaml_to_apply = item
            with open(yaml_to_apply) as f:
                complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
                for partial_yaml in complete_yaml:
                    if partial_yaml['kind'] == 'Deployment' or partial_yaml['kind'] == 'StatefulSet' or partial_yaml['kind'] == 'DaemonSet':
                        # update pod information
                        if s['regex']['edge-area']['pod'] == '':
                            s['regex']['edge-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            s['regex']['edge-area']['pod'] = f'{s['regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        
                        if metrics['global']['regex']['edge-area']['pod'] == '':
                            metrics['global']['regex']['edge-area']['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                        else:
                            metrics['global']['regex']['edge-area']['pod'] = f'{metrics['global']['regex']['edge-area']['pod']}|{partial_yaml['metadata']['name']}-.*'
                        # update workload information
                        if s['regex']['edge-area']['workload']['regex'] != '':
                            logger.error(f"Multiple deployments/statefulset/daemonset for the service {sc['name']} in the edge-area not supported")
                            exit(1)
                        if partial_yaml['kind'] == 'Deployment':
                            s['regex']['edge-area']['workload']['type'] == 'deployment'
                            s['regex']['edge-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                        if partial_yaml['kind'] == 'StatefulSet':
                            s['regex']['edge-area']['workload']['type'] == 'statefulset'
                            s['regex']['edge-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                        if partial_yaml['kind'] == 'DaemonSet':
                            s['regex']['edge-area']['workload']['type'] == 'daemonset'
                            s['regex']['edge-area']['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                                                                                                                                  
                        if metrics['global']['regex']['edge-area']['workload'] == '':
                            metrics['global']['regex']['edge-area']['workload'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global']['regex']['edge-area']['workload'] = f'{metrics['global']['regex']['edge-area']['workload']}|{partial_yaml['metadata']['name']}'
                    
                    if partial_yaml['kind'] == 'HorizontalPodAutoscaler' :
                        if s['regex']['edge-area']['hpa'] == '':
                            s['regex']['edge-area']['hpa'] = f'{partial_yaml['metadata']['name']}'
                            s['hpa']['edge-area']['min-replicas'] = partial_yaml['spec']['minReplicas']
                            s['hpa']['edge-area']['max-replicas'] = partial_yaml['spec']['maxReplicas']
                        else:
                            logger.error(f"Multiple HPA for the service {sc['name']} in the edge-area not supported")
                            exit(1)
                        if metrics['global']['regex']['edge-area']['hpa'] == '':
                            metrics['global']['regex']['edge-area']['hpa'] = f'{partial_yaml['metadata']['name']}'
                        else:
                            metrics['global']['regex']['edge-area']['hpa'] = f'{metrics['global']['regex']['edge-area']['hpa']}|{partial_yaml['metadata']['name']}'

def init():
    global gma_config, metrics, service_id_to_name

    logger.info(f"Init metrics dictionary")

    metrics = dict() # Global Status dictionary
    metrics['services'] = dict() # Services status dictionary
    
    # Initialize istio ingress  status information
    metrics[edge_istio_ingress_app]=dict() # Istio-ingress status dictionary

    # Initialize the service status information
    mid = 0 # microservice id
    services=metrics['services']
    for s in gma_config['spec']['services']:
            services[s['name']]=dict()
            # Initialize the service id
            if gma_config['spec']['explicit-service-id']:
                services[s['name']]['id'] = s['id']
                if s['id'] > mid:
                    mid = s['id']+1 # needed for istio-ingress id
            else:
                services[s['name']]['id'] = mid
                mid = mid +1
            
            # Initialize HPA dictionary
            services[s['name']]['hpa'] = dict()
            services[s['name']]['hpa']['cloud-area'] = dict()
            services[s['name']]['hpa']['cloud-area']['min-replicas'] = 0
            services[s['name']]['hpa']['cloud-area']['max-replicas'] = 0
            services[s['name']]['hpa']['cloud-area']['desired-replicas'] = 0
            services[s['name']]['hpa']['cloud-area']['current_replicas'] = 1    # default value, if no hpa is running the service is considdred to have 1 replica
            services[s['name']]['hpa']['cloud-area']['info'] = 'HPA status cloud-area'
            services[s['name']]['hpa']['cloud-area']['last_update'] = 0
            services[s['name']]['hpa']['edge-area'] = dict()
            services[s['name']]['hpa']['edge-area']['min-replicas'] = 0
            services[s['name']]['hpa']['edge-area']['max-replicas'] = 0
            services[s['name']]['hpa']['edge-area']['desired-replicas'] = 0
            services[s['name']]['hpa']['edge-area']['current_replicas'] = 1 # default value, if no hpa is running the service is considdred to have 1 replica
            services[s['name']]['hpa']['edge-area']['info'] = 'HPA status edge-area'
            services[s['name']]['hpa']['edge-area']['last_update'] = 0
            
            # Initialize the acpu values (actual cpu consumption)
            services[s['name']]['acpu'] = dict()
            services[s['name']]['acpu']['cloud-area'] = dict()
            services[s['name']]['acpu']['cloud-area']['value'] = 0
            services[s['name']]['acpu']['cloud-area']['last-update'] = 0
            services[s['name']]['acpu']['cloud-area']['info'] = 'CPU consumption in the cloud area in seconds per second'
            services[s['name']]['acpu']['edge-area'] = dict()
            services[s['name']]['acpu']['edge-area']['value'] = 0
            services[s['name']]['acpu']['edge-area']['last-update'] = 0
            services[s['name']]['acpu']['edge-area']['info'] = 'CPU consumption in the edge area in seconds per second'

            # Initialize the amem values (actuam memory consumption)
            services[s['name']]['amem'] = dict()
            services[s['name']]['amem']['cloud-area'] = dict()
            services[s['name']]['amem']['cloud-area']['info'] = 'Memory consumption in the cloud area in bytes'
            services[s['name']]['amem']['cloud-area']['value'] = 0
            services[s['name']]['amem']['cloud-area']['last-update'] = 0
            services[s['name']]['amem']['edge-area'] = dict()
            services[s['name']]['amem']['edge-area']['info'] = 'Memory consumption in the edge area in bytes'
            services[s['name']]['amem']['edge-area']['value'] = 0
            services[s['name']]['amem']['edge-area']['last-update'] = 0

            # # Initialize the replicas values
            # services[s['name']]['replicas'] = dict()
            # services[s['name']]['replicas']['cloud-area'] = dict()
            # services[s['name']]['replicas']['cloud-area']['value'] = 0
            # services[s['name']]['replicas']['cloud-area']['last-update'] = 0
            # services[s['name']]['replicas']['cloud-area']['info'] = 'N. replicas in the cloud area'
            # services[s['name']]['replicas']['edge-area'] = dict()
            # services[s['name']]['replicas']['edge-area']['value'] = 0
            # services[s['name']]['replicas']['edge-area']['last-update'] = 0
            # services[s['name']]['replicas']['edge-area']['info'] = 'N. replicas in the edge area'

            # Initialize the Rs values (response size)
            services[s['name']]['rs'] = dict()
            services[s['name']]['rs']['value'] = 0
            services[s['name']]['rs']['last-update'] = time.time()
            services[s['name']]['rs']['info'] = 'Response size in bytes'

            # Initialize the Fcm values (calling frequency matrix)
            services[s['name']]['fcm'] = dict()
            services[s['name']]['fcm']['value'] = dict()
            services[s['name']]['fcm']['last-update'] = 0
            services[s['name']]['fcm']['info'] = 'Call frequency matrix'

            # Initialize the request rate values (req/s)
            services[s['name']]['lambda'] = dict()
            services[s['name']]['lambda']['value'] = 0
            services[s['name']]['lambda']['last-update'] = 0
            services[s['name']]['lambda']['info'] = 'Service request rate req/s'

            # Inintialize the regex values for filtering pod, workload (deployment,statefulset,daemonset) and hpa from their names
            services[s['name']]['regex'] = dict()
            services[s['name']]['regex']['edge-area'] = dict()
            services[s['name']]['regex']['cloud-area'] = dict()
            services[s['name']]['regex']['edge-area']['pod'] = ''
            services[s['name']]['regex']['edge-area']['workload'] = dict()
            services[s['name']]['regex']['edge-area']['workload']['type'] = 'deployment'
            services[s['name']]['regex']['edge-area']['workload']['regex'] = ''
            services[s['name']]['regex']['edge-area']['hpa'] = ''
            services[s['name']]['regex']['cloud-area']['pod'] = ''
            services[s['name']]['regex']['cloud-area']['workload'] = dict()
            services[s['name']]['regex']['cloud-area']['workload']['type'] = 'deployment'
            services[s['name']]['regex']['cloud-area']['workload']['regex'] = ''
            services[s['name']]['regex']['cloud-area']['hpa'] = ''

            # Inintialize the yamls list
            services[s['name']]['instances'] = dict()
            services[s['name']]['instances']['cloud-yamls'] = s['instances']['cloud-yamls']
            services[s['name']]['instances']['edge-yamls'] = s['instances']['edge-yamls']

    # map service id to name
    for service_name in metrics['services']:
        service_id_to_name[metrics['services'][service_name]['id']] = service_name
    
    # Initialize Istio-ingress status
    metrics[edge_istio_ingress_app]['fcm'] = dict() # call frequency matrix
    metrics[edge_istio_ingress_app]['fcm']['info'] = 'Call frequency matrix'
    metrics[edge_istio_ingress_app]['fcm']['value'] = dict() 
    metrics[edge_istio_ingress_app]['fcm']['last-update'] = 0 # last update time
    metrics[edge_istio_ingress_app]['delay'] = dict() # average delay in milliseconds
    metrics[edge_istio_ingress_app]['delay']['value'] = 0
    metrics[edge_istio_ingress_app]['delay']['info'] = 'Average edge user delay in ms'
    metrics[edge_istio_ingress_app]['delay']['last-update'] = 0 # last update time
    metrics[edge_istio_ingress_app]['lambda'] = dict() # average delay in milliseconds
    metrics[edge_istio_ingress_app]['lambda']['value'] = 0
    metrics[edge_istio_ingress_app]['lambda']['info'] = 'Request rate from edge user in req/s'
    metrics[edge_istio_ingress_app]['lambda']['last-update'] = 0 # last update time

    # Initialize global status
    # Metrics arranged as a vector/matrix
    # value [i] is the metric of the service with id=i
    # Last id of vector/matrix is istio-ingress. The overall delay is the delay of istio-ingress
    metrics['global'] = dict()
    metrics['global']['regex'] = dict() # Global regex dictionary
    metrics['global']['regex']['edge-area'] = dict()
    metrics['global']['regex']['cloud-area'] = dict()
    metrics['global']['regex']['edge-area']['pod'] = ''
    metrics['global']['regex']['edge-area']['workload'] = ''
    metrics['global']['regex']['edge-area']['hpa'] = ''
    metrics['global']['regex']['cloud-area']['pod'] = ''
    metrics['global']['regex']['cloud-area']['workload'] = ''
    metrics['global']['regex']['cloud-area']['hpa'] = ''
    metrics['global']['n-services']= mid+1 # number of microservices
    M = metrics['global']['n-services'] 
    
    metrics['global']['fcm'] = dict()
    metrics['global']['fcm']['info'] = 'Call frequency matrix'
    metrics['global']['fcm']['value'] = np.zeros((M,M))
    metrics['global']['fcm']['last-update'] = 0 # last update time

    metrics['global']['rs'] = dict()
    metrics['global']['rs']['info'] = 'Response size vector in bytes'
    metrics['global']['rs']['value'] = np.zeros(M)
    metrics['global']['rs']['last-update'] = 0 # last update time

    metrics['global']['replicas'] = dict()
    metrics['global']['replicas']['cloud-area'] = dict()
    metrics['global']['replicas']['cloud-area']['info'] = 'Replicas vector for cloud area'
    metrics['global']['replicas']['cloud-area']['value'] = np.zeros(M)
    metrics['global']['replicas']['cloud-area']['last-update'] = 0 # last update time
    metrics['global']['replicas']['edge-area'] = dict()
    metrics['global']['replicas']['edge-area']['info'] = 'Replicas vector for edge area'
    metrics['global']['replicas']['edge-area']['value'] = np.zeros(M)
    metrics['global']['replicas']['edge-area']['last-update'] = 0 # last update time
    
    metrics['global']['acpu'] = dict()
    metrics['global']['acpu']['cloud-area'] = dict()
    metrics['global']['acpu']['cloud-area']['info'] = 'Actual CPU utilizatiion vector in seconds per second for cloud area'
    metrics['global']['acpu']['cloud-area']['value'] = np.zeros(M)
    metrics['global']['acpu']['cloud-area']['last-update'] = 0 # last update time
    metrics['global']['acpu']['edge-area'] = dict()
    metrics['global']['acpu']['edge-area']['info'] = 'Actual CPU utilizatiion vector in seconds per second for cloud area'
    metrics['global']['acpu']['edge-area']['value'] = np.zeros(M)
    metrics['global']['acpu']['edge-area']['last-update'] = 0 # last update time
    
    metrics['global']['amem'] = dict()
    metrics['global']['amem']['cloud-area'] = dict()
    metrics['global']['amem']['cloud-area']['info'] = 'Actual memory utilizatiion vector in bytes for cloud area'
    metrics['global']['amem']['cloud-area']['value'] = np.zeros(M)
    metrics['global']['amem']['cloud-area']['last-update'] = 0 # last update time
    metrics['global']['amem']['edge-area'] = dict()
    metrics['global']['amem']['edge-area']['info'] = 'Actual memory utilizatiion vector in bytes for edge area'
    metrics['global']['amem']['edge-area']['value'] = np.zeros(M)
    metrics['global']['amem']['edge-area']['last-update'] = 0 # last update time
    
    metrics['global']['lambda'] = dict()   
    metrics['global']['lambda']['info'] = 'Request rate vector in req/s'
    metrics['global']['lambda']['value'] = np.zeros(M)
    metrics['global']['lambda']['last-update'] = 0 # last update time
    
    # delay of HTTP GET to services
    metrics['global']['delay'] = dict()
    metrics['global']['delay']['cloud-area'] = dict()
    metrics['global']['delay']['cloud-area']['value'] = np.zeros(M)
    metrics['global']['delay']['cloud-area']['info'] = 'Average delay vector in ms'
    metrics['global']['delay']['cloud-area']['last-update'] = 0 # last update time
    metrics['global']['delay']['edge-area'] = dict()
    metrics['global']['delay']['edge-area']['value'] = np.zeros(M)
    metrics['global']['delay']['edge-area']['info'] = 'Average delay vector in ms'
    metrics['global']['delay']['edge-area']['last-update'] = 0 # last update time
    
    # target delay for HTTP GET to services
    metrics['global']['target-delay'] = dict()
    metrics['global']['target-delay']['cloud-area'] = dict()
    metrics['global']['target-delay']['cloud-area']['value'] = np.zeros(M)
    metrics['global']['target-delay']['cloud-area']['info'] = 'Average target-delay vector in ms'
    metrics['global']['target-delay']['cloud-area']['last-update'] = 0 # last update time
    metrics['global']['target-delay']['edge-area'] = dict()
    metrics['global']['target-delay']['edge-area']['value'] = np.zeros(M)
    metrics['global']['target-delay']['edge-area']['info'] = 'Average target-delay vector in ms'
    metrics['global']['target-delay']['edge-area']['last-update'] = 0 # last update time
    
    metrics['global']['rtt'] = dict()
    metrics['global']['rtt']['cloud-area'] = dict()
    metrics['global']['rtt']['cloud-area']['value'] = time_to_ms_converter(gma_config['spec']['cloud-area']['rtt'])
    metrics['global']['rtt']['cloud-area']['info'] = 'Round trip time to reach the cloud area in ms'
    metrics['global']['rtt']['edge-area'] = dict()
    metrics['global']['rtt']['edge-area']['value'] = time_to_ms_converter(gma_config['spec']['edge-area']['rtt'])
    metrics['global']['rtt']['edge-area']['info'] = 'Round trip time to reach the edge area in ms'

    metrics['global']['network-capacity'] = dict()
    metrics['global']['network-capacity']['edge-area'] = dict()
    metrics['global']['network-capacity']['edge-area']['value'] = bitrate_to_bps_converter(gma_config['spec']['edge-area']['network-capacity'])
    metrics['global']['network-capacity']['edge-area']['info'] = 'Network capacity in bit per second to reach the edge area'
    metrics['global']['network-capacity']['cloud-area'] = dict()
    metrics['global']['network-capacity']['cloud-area']['value']=bitrate_to_bps_converter(gma_config['spec']['cloud-area']['network-capacity'])
    metrics['global']['network-capacity']['cloud-area']['info'] = 'Network capacity in bit per second between to reach the cloud area'
    
    metrics['global']['cost'] = dict()
    metrics['global']['cost']['edge-area'] = dict()
    metrics['global']['cost']['edge-area']['cpu'] = dict()
    metrics['global']['cost']['edge-area']['cpu']['value'] = gma_config['spec']['edge-area']['cost']['cpu']
    metrics['global']['cost']['edge-area']['cpu']['info'] = 'Cost of CPU in the edge area'
    metrics['global']['cost']['edge-area']['memory'] = dict()
    metrics['global']['cost']['edge-area']['memory']['value'] = gma_config['spec']['edge-area']['cost']['memory']
    metrics['global']['cost']['edge-area']['memory']['info'] = 'Cost of memory in the edge area'
    metrics['global']['cost']['cloud-area'] = dict()
    metrics['global']['cost']['cloud-area']['cpu'] = dict()
    metrics['global']['cost']['cloud-area']['cpu']['value'] = gma_config['spec']['cloud-area']['cost']['cpu']
    metrics['global']['cost']['cloud-area']['cpu']['info'] = 'Cost of CPU in the edge area'
    metrics['global']['cost']['cloud-area']['memory'] = dict()
    metrics['global']['cost']['cloud-area']['memory']['value'] = gma_config['spec']['cloud-area']['cost']['memory']
    metrics['global']['cost']['cloud-area']['memory']['info'] = 'Cost of memory in the edge area'

    # Get the pod/deployment regex for each service
    parse_yaml()
    check_inits()

def check_inits():
    # check workload exists for any service
    for service_name in metrics['services']:
        if metrics['services'][service_name]['regex']['cloud-area']['workload']['regex'] == '':
            logger.error(f"Workload not found for service {service_name} in the cloud-area")
            exit(1)
        if metrics['services'][service_name]['regex']['edge-area']['workload']['regex'] == '':
            logger.error(f"Workload not found for service {service_name} in the edge-area")
            exit(1)
    # check workloads for cloud area and edge have the same type
    for service_name in metrics['services']:
        if metrics['services'][service_name]['regex']['cloud-area']['workload']['type'] != metrics['services'][service_name]['regex']['edge-area']['workload']['type']:
            logger.error(f"Workload type for service {service_name} in the cloud-area and edge-area are different")
            exit(1)
    # check hpa exists for any service
    for service_name in metrics['services']:
        if metrics['services'][service_name]['regex']['cloud-area']['hpa'] == '':
            logger.error(f"HPA not found for service {service_name} in the cloud-area")
            exit(1)
        if metrics['services'][service_name]['regex']['edge-area']['hpa'] == '':
            logger.error(f"HPA not found for service {service_name} in the edge-area")
            exit(1)
    #TODO check the node of the cluster have different topology labels

    

def time_to_ms_converter(delay_string):
    delay_string = str(delay_string)
    if delay_string.endswith("ms"):
        value = int(delay_string.split("ms")[0]) 
    elif delay_string.endswith("s"):
        value = int(delay_string.split("s")[0])*1000
    elif delay_string.endswith("m"):
        value = int(delay_string.split("m")[0])*60000
    elif delay_string.endswith("h"):    
        value = int(delay_string.split("h")[0])*3600000
    else:
        value = int(delay_string)*1000
    return value

def bitrate_to_bps_converter(cap_string):
    cap_string=str(cap_string)
    if cap_string.endswith("kbps"):
        value = int(cap_string.split("kbps")[0])*1000 
    elif cap_string.endswith("Mbps"):
        value = int(cap_string.split("Mbps")[0])*1000000
    elif cap_string.endswith("Gbps"):
        value = int(cap_string.split("Gbps")[0])*1000000000
    elif cap_string.endswith("bps"):   
        value = int(cap_string.split("bps")[0])
    else:
        value = int(cap_string)
    return value

class GMAStataMachine():
    # hpa_running: HPA is runnning
    # camping: periodic monitoring of the system and no need to take action
    # offload_alarm: offload delay threshold is reached; check if this state persist for a while
    # unoffload_alarm: unoffload delay threshold is reached; check if this state persist for a while
    # offloading: offload in progress
    # unoffloading: unoffload in progress

    def __init__(self):
        self.run()
        return

    def hpa_running(self):
        logger.info('_________________________________________________________')
        logger.info('Entering HPA Running')
        time.sleep(stabilizaiton_window_sec)
        if update_and_check_HPA():
            self.next = self.hpa_running
        else:
            self.next = self.camping
        return

    def camping(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Camping State')
        if update_and_check_HPA():
            self.next = self.hpa_running
            return
        update_delay()
        logger.info(f'user delay: {metrics['global']['delay']['edge-area']['value'][M-1]}')
        if metrics['global']['delay']['edge-area']['value'][M-1] > offload_delay_threshold_ms:
            if np.all(metrics['global']['replicas']['edge-area']['value'][:-1] > 0):
                logger.warning('All microservice in the edge area, can not offload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sec')
                time.sleep(sync_period_sec)     
            else:
                self.next = self.offload_alarm
            return
        elif metrics['global']['delay']['edge-area']['value'][M-1] < unoffload_delay_threshold_ms:
            logger.info('Delay below unoffload threshold')
            if np.all(metrics['global']['replicas']['edge-area']['value'][:-1] == 0):
                logger.warning('No microservice in the edge area, can not unoffload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sec')
                time.sleep(sync_period_sec) 
            else:
                self.next = self.unoffload_alarm
            return
        else:
            self.next = self.camping
            logger.info(f'sleeping for {sync_period_sec} sec')
            time.sleep(sync_period_sec)
        return
        
    def offload_alarm(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Offload Alarm')
        stabilization_cycle_sec = 30
        n_cycles = int(np.ceil(stabilizaiton_window_sec/stabilization_cycle_sec))
        for i in range(n_cycles):
            if update_and_check_HPA():
                self.next = self.hpa_running
                return
            update_delay()
            logger.info(f'user delay: {metrics['global']['delay']['edge-area']['value'][M-1]}')
            if metrics['global']['delay']['edge-area']['value'][M-1] > offload_delay_threshold_ms:
                logger.info(f'sleeping for {stabilizaiton_window_sec-i*stabilization_cycle_sec} sec')
                time.sleep(stabilization_cycle_sec)
            else:
                self.next = self.camping
                return
        self.next = self.offloading
        return
    
    def unoffload_alarm(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Unoffload Alarm')
        stabilization_cycle_sec = 30
        n_cycles = int(np.ceil(stabilizaiton_window_sec/stabilization_cycle_sec))
        for i in range(n_cycles):
            if update_and_check_HPA():
                self.next = self.hpa_running
                return
            update_delay()
            logger.info(f'user delay: {metrics['global']['delay']['edge-area']['value'][M-1]}')
            if metrics['global']['delay']['edge-area']['value'][M-1] < unoffload_delay_threshold_ms:
                logger.info(f'sleeping for {stabilizaiton_window_sec-i*stabilization_cycle_sec} sec')
                time.sleep(stabilization_cycle_sec)
            else:
                self.next = self.camping
                return
        self.next = self.unoffloading
        return
    
    def offloading(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Offloading')
        update_metrics()
        metric_to_pass = metrics['global'].copy()
        metric_to_pass['target-delay']['edge-area']['value'][-1] = offload_delay_threshold_ms
        params = SBMP_GMA_Connector.Connector(metric_to_pass)
        # offloading logic
        result_list = EPAMP_offload_caching.offload(params)
        logger.info(f"Offload result: {result_list[1]['info']}")
        apply_configuration(result_list)
        logger.info(f'sleeping for {stabilizaiton_window_sec} sec')
        time.sleep(stabilizaiton_window_sec)
        # offloading done
        self.next = self.camping
        return
    
    def unoffloading(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Unoffloading')
        update_metrics()
        metric_to_pass = metrics['global'].copy()
        metric_to_pass['target-delay']['edge-area']['value'][-1] = offload_delay_threshold_ms
        params = SBMP_GMA_Connector.Connector(metric_to_pass)
        # unoffloading logic
        result_list = EPAMP_unoffload_from_void.unoffload(params)
        logger.info(f"Unoffload result: {result_list[1]['info']}")
        apply_configuration(result_list)
        logger.info(f'sleeping for {stabilizaiton_window_sec} sec')
        time.sleep(stabilizaiton_window_sec)
        # unoffloading done
        self.next = self.camping
        return
    
    def run(self):
        self.next = self.camping
        while True:
            self.next()

# Main function
if __name__ == "__main__":
    config_env=environ.get('GPA_CONFIG', './GMAConfig.yaml')
    log_env=environ.get('GPA_LOG_LEVEL', 'INFO')

    parser = argparse.ArgumentParser()
    parser.add_argument( '-c',
                     '--configfile',
                     default='',
                     help='Provide config file. Example --configfile ./GMAConfig.yaml, default=./GMAConfig.yaml' )
    parser.add_argument( '-log',
                     '--loglevel',
                     default='',
                     help='Provide logging level. Example --loglevel debug, default=warning')

    args = parser.parse_args()

    if args.loglevel == '':
        args.loglevel = log_env
    if args.configfile == '':
        args.configfile = config_env
    
    # Set up logger
    logging.basicConfig(stream=sys.stdout, level=args.loglevel.upper(),format='%(asctime)s GMA %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    logger_stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(logger_stream_handler)
    logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s GMA %(levelname)s %(message)s'))
    logger.propagate=False
    logger.info(f"Starting GMA with config file: {args.configfile}")
    

    # Load the configuration YAML file
    try:
        with open(args.configfile, 'r') as file:
            yaml_data = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.configfile}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        logger.error(f"Error in configuration file: {exc}")
        sys.exit(1)
    
    # Initialize metrics dict and service to id mapping
    metrics = dict()
    service_id_to_name = dict()
    
    # Convert the YAML data to a dictionary
    gma_config = dict(yaml_data)
    
    # Initialize Kubernetes clients
    k8s_apiclient = dict()
    if gma_config['spec']['cloud-area']['context']!='':
        k8s_apiclient['cloud-area'] = kubernetes.config.kube_config.new_client_from_config(context=gma_config['spec']['cloud-area']['context'])
    else:
        k8s_apiclient['cloud-area'] = kubernetes.config.kube_config.new_client_from_config()
        
    if gma_config['spec']['edge-area']['context']!='':
        k8s_apiclient['edge-area'] = kubernetes.config.kube_config.new_client_from_config(context=gma_config['spec']['edge-area']['context'])
    else:
        k8s_apiclient['edge-area'] = kubernetes.config.kube_config.new_client_from_config()
    
    # Create a Prometheus client
    try:
        prom_client = PrometheusConnect(url=gma_config['spec']['prometheus-url'], disable_ssl=True)
    except PrometheusApiClientException:
        logger.error(f"Error connecting to Prometheus server: {gma_config['spec']['prometheus-url']}")
        sys.exit(1)

    # Initialize the microservice metrics dictionary
    init()

    # global variables short-names
    sync_period_sec = time_to_ms_converter(gma_config['spec']['sync-period'])/1000
    query_period_str = gma_config['spec']['query-period']
    stabilizaiton_window_sec = time_to_ms_converter(gma_config['spec']['stabilization-window'])/1000
    offload_delay_threshold_ms = time_to_ms_converter(gma_config['spec']['offload-delay-threshold'])
    unoffload_delay_threshold_ms = time_to_ms_converter(gma_config['spec']['unoffload-delay-threshold'])
    M =  metrics['global']['n-services']  # number of microservices
    edge-istio-ingress-app = metrics['services'][gma_config['spec']['edge-area']['istio-ingress-source-app']]['id']


    # Run the state machine
    sm = GMAStataMachine()