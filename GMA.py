import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_dir}/utils')
sys.path.append(f'{current_dir}/strategies')

import argparse
import logging
import re
import sys
import time
import subprocess
import requests


import kubernetes
import numpy as np
import yaml

import importlib
#import SBMP_GMA_Connector

from os import environ
from prometheus_api_client import PrometheusConnect,PrometheusApiClientException



def update_ucpu():
    global gma_config, prom_client, metrics

    logger.info(f"Update actual cpu utilization")

    now = time.time()

    services=status['service-info']
    
    # update values
    for area in areas:
        pod_regex = status['global-regex'][area]['pod']
        query_cpu = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{cluster="{cluster[area]}",namespace="{namespace}",pod=~"{pod_regex}",container!="istio-proxy",container!=""}}[{query_period_str}]))'
        try:
            query_results = prom_client.custom_query(query=query_cpu)
        except PrometheusApiClientException as e:
            logger.error(f"Prometheus query exception for query {query_cpu}: {str(e)}")
            return
        
        # clean the ucpu values
        status['service-metrics']['ucpu'][area]['value'] = np.zeros(M, dtype=float)
        status['service-metrics']['ucpu'][area]['last-update'] = now
        
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        if result["value"][1]=="NaN":
                            value = 0
                        else:
                            value = float(result["value"][1])
                        status['service-metrics']['ucpu'][area]['value'][service['id']] = status['service-metrics']['ucpu'][area]['value'][service['id']]+value

def update_umem():
    global gma_config, prom_client, metrics

    logger.info(f"Update actual memory utilization")

    now = time.time()

    services=status['service-info']
    
    # update values
    for area in areas:
        pod_regex = status['global-regex'][area]['pod']
        query_mem = f'sum by (pod) (container_memory_usage_bytes{{cluster="{cluster[area]}", namespace="{namespace}",pod=~"{pod_regex}",container!="istio-proxy",container!=""}})'
        try:
            query_results = prom_client.custom_query(query=query_mem)
        except PrometheusApiClientException as e:
            logger.error(f"Prometheus query exception for query {query_mem}: {str(e)}")
            return
        
        status['service-metrics']['umem'][area]['value'] = np.zeros(M, dtype=float)
        status['service-metrics']['umem'][area]['last-update'] = now
        if query_results:
            for result in query_results:
                for service_name in services:
                    service=services[service_name]
                    if re.search(service['regex'][area]['pod'], result['metric']['pod'], re.IGNORECASE):
                        if result["value"][1]=="NaN":
                            value = 0
                        else:
                            value = float(result["value"][1])
                        status['service-metrics']['umem'][area]['value'][service['id']] = status['service-metrics']['umem'][area]['value'][service['id']] + value

def update_ingress_lambda():
    global gma_config, prom_client, status, M

    logger.info(f"Update request rate values")

    now = time.time()

    # update lambda values
    destination_app_regex = "|".join(status['service-info'].keys())
    query_lambda = f'sum by (source_app) (rate(istio_requests_total{{cluster="{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200", instance=~"{edge_pod_cidr_regex}"}}[{query_period_str}]))'
    try:
        query_result = prom_client.custom_query(query=query_lambda)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {query_lambda}: {str(e)}")
        return
    
    #clean ingress lambda values
    status['service-metrics']['service-lambda']['value'][M-1] = 0 # M-1 is the index of the istio-ingress in the global lambda vector
    status['service-metrics']['service-lambda']['last-update'] = now
    
    lambda_edge = 0
    if query_result:
        for result in query_result:
            if result["value"][1]=="NaN":
                lambda_edge = 0
            else:
                lambda_edge = float(result["value"][1])
            if status['service-metrics']['service-lambda']['value'][M-1] != 0:
                logger.critical(f"Multiple results for the lambda query {query_lambda} and service {edge_istio_ingress_app}")
                exit(1)
            status['service-metrics']['service-lambda']['value'][M-1] = lambda_edge # M-1 is the index of the istio-ingress in the global lambda vector
            break
    
    # update resource scaling used for multi edge environment
    query_lambda_tot = f'sum by (source_app) (rate(istio_requests_total{{namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}]))'
    try:
        query_result = prom_client.custom_query(query=query_lambda_tot)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {query_lambda_tot}: {str(e)}")
        return
    
    #clean ingress lambda values

    if query_result:
        if len(query_result) > 1:
            logger.critical(f"Multiple results for the lambda tot query {query_lambda_tot} and service {edge_istio_ingress_app}")
            exit(1)
        for result in query_result:
            if result["value"][1]=="NaN":
                lambda_tot = 0
            else:
                lambda_tot = float(result["value"][1])
            if lambda_tot > 0:
                me_resource_scaling = lambda_edge / lambda_tot   
                status['service-metrics']['me-resource-scaling']['value'] = np.ones(M,dtype=float)*me_resource_scaling
            break
    return


def update_response_length():
    global gma_config, prom_client, metrics

    logger.info(f"Update response size values")

    now = time.time()

    if status['service-metrics']['service-lambda']['value'][M-1] == 0:
        logger.info(f"Lambda value for the istio-ingress is 0, skipping Rs update")
        return

    # update Rs values
    destination_app_regex = "|".join(status['service-info'].keys())
    cluster_regex = cluster['cloud-area']+"|"+cluster['edge-area'] 
    query_Rs = f'sum by (destination_app) (increase(istio_response_bytes_sum{{cluster=~"{cluster_regex}",namespace="{namespace}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))/sum by (destination_app) (increase(istio_response_bytes_count{{cluster=~"{cluster_regex}",namespace="{namespace}", response_code="200", destination_app=~"{destination_app_regex}", reporter="destination"}}[{query_period_str}]))'
    try:
        r1 = prom_client.custom_query(query=query_Rs)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {query_Rs}: {str(e)}")
        return
    
    # clean Rs values
    status['service-metrics']['response-length']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['response-length']['last-update'] = now
    
    if r1:
        for result in r1:
            service_name = result["metric"]["destination_app"]
            if service_name in status['service-info']:
                service=status['service-info'][service_name]
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])
                if status['service-metrics']['response-length']['value'][service['id']] != 0:
                    logger.critical(f"Multiple results for the Rs query {query_Rs} and service {service_name}")
                    exit(1)
                status['service-metrics']['response-length']['value'][service['id']] = value
                status['service-metrics']['response-length']['last-update'] = now
    return

def update_Fm_and_lambda():
    global gma_config, prom_client, metrics
    
    logger.info(f"Update call frequency matrix")

    now = time.time()

    if status['service-metrics']['service-lambda']['value'][M-1] == 0:
        logger.info(f"Lambda value for the istio-ingress is 0, skipping Fm and lambda update")
        return

    # update lambda and Fm values
    destination_app_regex = "|".join(status['service-info'].keys())
    cluster_regex = cluster['cloud-area']+"|"+cluster['edge-area']
    source_app_regex = edge_istio_ingress_app+"|"+destination_app_regex
    fm_query_num=f'sum by (source_app,destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{namespace}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    lambda_query=f'sum by (destination_app) (rate(istio_requests_total{{cluster=~"{cluster_regex}",namespace="{namespace}",source_app=~"{source_app_regex}",destination_app=~"{destination_app_regex}",reporter="destination",response_code="200"}}[{query_period_str}])) '
    try:
        r = prom_client.custom_query(query=lambda_query)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {lambda_query}: {str(e)}")
        return
    try:
        r2 = prom_client.custom_query(query=fm_query_num)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {fm_query_num}: {str(e)}")
        return
    
    # clean lambda and Fm values
    status['service-metrics']['fm']['value'] = np.zeros((M,M),dtype=float)
    status['service-metrics']['fm']['last-update'] = now
    status['service-metrics']['service-lambda']['value'][:M-1] = 0 # M-1 is the index of the istio-ingress in the global lambda vector and this function does not update the lambda of the istio-ingress due to the reporter="destination" filter. update_ingress_lambda() function is responsible for that.
    status['service-metrics']['service-lambda']['last-update'] = now
    
    for result in r:
        destination_service_name = result["metric"]["destination_app"]
        if destination_service_name in status['service-info']:
            destination_service = status['service-info'][destination_service_name]
            if result["value"][1]=="NaN":
                value = 0
            else:
                value = float(result["value"][1])
            if status['service-metrics']['service-lambda']['value'][destination_service['id']] != 0:
                logger.critical(f"Multiple results for the lambda query {lambda_query} and service {destination_service_name}")
                exit(1)
            status['service-metrics']['service-lambda']['value'][destination_service['id']] = value
            status['service-metrics']['service-lambda']['last-update'] = now
            continue

    for result in r2:
        source_service_name = result["metric"]["source_app"]
        destination_service_name = result["metric"]["destination_app"]
        if source_service_name in status['service-info'] and destination_service_name in status['service-info']:
            destination_service = status['service-info'][destination_service_name]
            source_service = status['service-info'][source_service_name]
            if status['service-metrics']['service-lambda']['value'][source_service['id']] == 0:
                value = 0
            else:
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])/status['service-metrics']['service-lambda']['value'][source_service['id']]

            if status['service-metrics']['fm']['value'][source_service['id']][destination_service['id']]!=0:
                logger.critical(f"Multiple results for the Fm query {fm_query_num} and source service {source_service_name}, destination service {destination_service_name}")
                exit(1)
            status['service-metrics']['fm']['value'][source_service['id']][destination_service['id']] = value
            status['service-metrics']['fm']['last-update'] = now
            continue
        if source_service_name == edge_istio_ingress_app and destination_service_name in status['service-info']:
            if status['service-metrics']['service-lambda']['value'][M-1] == 0: # M-1 is the index of the istio-ingress in the global lambda vector
                value = 0
            else:
                if result["value"][1]=="NaN":
                    value = 0
                else:
                    value = float(result["value"][1])/status['service-metrics']['service-lambda']['value'][M-1]
            if status['service-metrics']['fm']['value'][M-1][status['service-info'][destination_service_name]['id']] != 0:
                logger.critical(f"Multiple results for the Fm query {fm_query_num} and source service {source_service_name}, destination service {destination_service_name}")
                exit(1)
            status['service-metrics']['fm']['value'][M-1][status['service-info'][destination_service_name]['id']] = value
            status['service-metrics']['fm']['last-update'] = now
    return

# Function that get the average delay from the istio-ingress gateway
def update_ingress_delay():
    global gma_config, prom_client, metrics

    logger.info(f"Update delay values from istio ingress in the edge area")

    now = time.time()

    # update the delay value
    destination_app_regex = "|".join(status['service-info'].keys())
    
    query_avg_delay = f'sum by (source_app) (rate(istio_request_duration_milliseconds_sum{{cluster=~"{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}])) / sum by (source_app) (rate(istio_request_duration_milliseconds_count{{cluster=~"{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}]))'
    try:
        result_query = prom_client.custom_query(query=query_avg_delay)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {query_avg_delay}: {str(e)}")
        return
    
    # clean the delay value
    status['service-metrics']['edge-user-delay']['value'] = 0
    status['service-metrics']['edge-user-delay']['last-update'] = now

    if result_query:
        for result in result_query:
            if result["value"][1]=="NaN":
                value=0
            else:
                value=float(result["value"][1])
            if status['service-metrics']['edge-user-delay']['value'] != 0:
                logger.critical(f"Multiple results for the delay query {query_avg_delay} and service {edge_istio_ingress_app}")
                exit(1)
            status['service-metrics']['edge-user-delay']['value'] = value
            status['service-metrics']['edge-user-delay']['last-update'] = now

def update_ingress_delay_quantile():
    global gma_config, prom_client, metrics

    logger.info(f"Update delay quantile values from istio ingress in the edge area")

    now = time.time()

    # update the delay quantile value
    destination_app_regex = "|".join(status['service-info'].keys())
    
    # weigthed average of the delay quantiles from istio-ingress to contacted microservices
    query_quantile_delay = f'sum(histogram_quantile({delay_quantile}, sum by (destination_app,le) (rate(istio_request_duration_milliseconds_bucket{{cluster=~"{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200"}}[{query_period_str}])))*(sum by (destination_app) (rate(istio_request_duration_milliseconds_bucket{{cluster=~"{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200",le="+Inf"}}[{query_period_str}])))) / scalar(sum(rate(istio_request_duration_milliseconds_bucket{{cluster=~"{cluster['edge-area']}", namespace="{edge_istio_ingress_namespace}", source_app="{edge_istio_ingress_app}", destination_app=~"{destination_app_regex}", reporter="source", response_code="200",le="+Inf"}}[{query_period_str}])))'
    try:
        result_query = prom_client.custom_query(query=query_quantile_delay)
    except PrometheusApiClientException as e:
        logger.error(f"Prometheus query exception for query {query_quantile_delay}: {str(e)}")
        return
    
    # clean the delay value
    status['service-metrics']['edge-user-delay-quantile']['value'] = 0
    status['service-metrics']['edge-user-delay-quantile']['last-update'] = now

    if result_query:
        for result in result_query:
            if result["value"][1]=="NaN":
                value=0
            else:
                value=float(result["value"][1])
            if status['service-metrics']['edge-user-delay-quantile']['value'] != 0:
                logger.critical(f"Multiple results for the delay quantile query {query_quantile_delay}")
                exit(1)
            status['service-metrics']['edge-user-delay-quantile']['value'] = value

# Function that updates the HPA values
def update_and_check_HPA():
    global gma_config, prom_client, status, k8s_apiclient

    logger.info(f"Update HPA values")
    now = time.time()

    hpa_running = False
    
    # update the hpa values
    for area in areas:
        with k8s_apiclient[area] as api_client:
        # Create an instance of the API class
            api_instance = kubernetes.client.AutoscalingV1Api(api_client)
            try:
                api_response = api_instance.list_namespaced_horizontal_pod_autoscaler(namespace,pretty='True',)
            except Exception as e:
                print("Exception when calling AutoscalingV1Api->list_namespaced_horizontal_pod_autoscaler: %s\n" % e)
                return
            
            # clean values
            status['service-metrics']['hpa'][area]['current-replicas'] = np.zeros(M, dtype=int)
            status['service-metrics']['hpa'][area]['desired-replicas'] = np.zeros(M, dtype=int)
            status['service-metrics']['hpa'][area]['last-update'] = now
            # no check for the istio-ingress
            status['service-metrics']['hpa']['edge-area']['current-replicas'][M-1] = 1 # the istio-ingress is always running on the edge area and the number of replicas do not matter but must be set to 1 as current-replicas is used to compute the current position of microservices
            status['service-metrics']['hpa']['edge-area']['desired-replicas'][M-1] = 1
            
            for hpa in api_response.items:
                if re.search(status['global-regex'][area]['hpa'], hpa.metadata.name, re.IGNORECASE):
                    for service_name in status['service-info']:
                        service=status['service-info'][service_name]
                        if re.search(service['regex'][area]['hpa'], hpa.metadata.name, re.IGNORECASE):
                            status['service-metrics']['hpa'][area]['current-replicas'][service['id']] = int(hpa.status.current_replicas)
                            status['service-metrics']['hpa'][area]['desired-replicas'][service['id']] = int(hpa.status.desired_replicas)
                            status['service-metrics']['hpa'][area]['last-update'] = now
                            run_cond1 = hpa.status.desired_replicas!=hpa.status.current_replicas and hpa.status.desired_replicas<=status['service-metrics']['hpa'][area]['max-replicas'][service['id']] and hpa.status.desired_replicas>=status['service-metrics']['hpa'][area]['min-replicas'][service['id']]
                            run_cond2 = status['service-metrics']['hpa'][area]['old-current-replicas'][service['id']]!=hpa.status.current_replicas and status['service-metrics']['hpa'][area]['old-current-replicas'][service['id']]>0     
                            if run_cond1 or run_cond2:
                                hpa_running = hpa_running or True
                                logging.info(f"HPA {hpa.metadata.name} for service {service_name} in {area} area is possibly running")
                            status['service-metrics']['hpa'][area]['old-current-replicas'][service['id']] = hpa.status.current_replicas
                            break
    return hpa_running

# Function that updates all metrics
def update_full_metrics():
    update_ucpu()
    update_umem()
    update_ingress_lambda()
    update_response_length()
    update_Fm_and_lambda()
    update_ingress_delay()
    update_ingress_delay_quantile()
    update_net_metrics()
    return

def apply_configuration(result_list):
    global gma_config, status, service_id_to_name, gma_config
    result_cloud_area = result_list[0] # result_list[0] contains cloud-area information
    result_edge_area = result_list[1] # result_list[1] contains edge-area information
    
    # remove resources from edge area
    for service_id in result_edge_area['to-delete']:
        if service_id not in service_id_to_name:
            continue
        name = service_id_to_name[service_id]
        service=status['service-info'][name]
        
        # move back replicas to cloud area
        workload_name = service['regex']['cloud-area']['workload']['regex']
        workload_type = service['regex']['cloud-area']['workload']['type']
        if workload_type != 'daemonset':
            cloud_replicas_increase = np.ceil(status['service-metrics']['hpa']['edge-area']['current-replicas'][service_id]/status['service-metrics']['me-resource-scaling']['value'][service_id]) #-status['service-metrics']['hpa']['cloud-area']['current-replicas'][service_id])
            cloud_replicas = status['service-metrics']['hpa']['cloud-area']['current-replicas'][service_id]+cloud_replicas_increase
            cloud_replicas = min(status['service-metrics']['hpa']['cloud-area']['max-replicas'][service_id],cloud_replicas)
            cloud_replicas = max(status['service-metrics']['hpa']['cloud-area']['min-replicas'][service_id],cloud_replicas)
            command = f'kubectl --context {gma_config['spec']['cloud-area']['context']} -n {namespace} scale {workload_type} {workload_name} --replicas {int(cloud_replicas)}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Scale {workload_type} {workload_name} in cloud-area to {cloud_replicas} replicas: {output}")            
        
        # delete resources in edge area
        for files in service['instances']['edge-yamls']:
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {namespace} delete -f {files}'
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
        service=status['service-info'][name]
        for files in service['instances']['edge-yamls']:
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {namespace} apply -f {files}'
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
            edge_replicas = np.ceil(status['service-metrics']['me-resource-scaling']['value'][service_id] * status['service-metrics']['hpa']['cloud-area']['current-replicas'][service_id])
            edge_replicas = min(status['service-metrics']['hpa']['edge-area']['max-replicas'][service_id],edge_replicas)
            edge_replicas = max(status['service-metrics']['hpa']['edge-area']['min-replicas'][service_id],edge_replicas)
            command = f'kubectl --context {gma_config['spec']['edge-area']['context']} -n {namespace} scale {workload_type} {workload_name} --replicas {int(edge_replicas)}'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.output
                # Handle the exception or log the error message
            logger.info(f"Scale deployment {service['regex']['edge-area']['workload']['regex']} in edge-area to {edge_replicas} replicas: {output}")

def update_net_metrics():
    logger.info(f"Update net metrics")
    netinfo_file = gma_config['spec']['network']['netinfo-file']
    net_prober_url = gma_config['spec']['network']['net-prober-url']
    if netinfo_file == '':
        logger.info(f"Netinfo file not properly configured in the configuration file, no net metric update performed")
        return
    
    with open(netinfo_file) as f:
            complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
            for partial_yaml in complete_yaml:
                if partial_yaml['kind'] == 'NetInfo':
                    netinfo = partial_yaml
                    break
    
    # update netinfo file with the net probing results
    if net_prober_url != '':
        logger.info(f"Net probing through {net_prober_url} ")
        try:
            response = requests.get(net_prober_url)
            netinfop = response.json()
            netinfo['spec']['edge-cloud-rtt'] = f'{int(netinfop['edge-cloud-rtt'])}ms'
            netinfo['spec']['cloud-edge-bps'] = f'{int(netinfop['cloud-edge-bps']/1e6)}Mbps'
            netinfo['spec']['edge-cloud-bps'] = f'{int(netinfop['edge-cloud-bps']/1e6)}Mbps'
            # write the netinfo to the file netinfo_file
            with open(netinfo_file, 'w') as f:
                yaml.dump(netinfo, f)
        except Exception as e:
            logger.error(f"Net probing failed: {str(e)}")
    
    if 'spec' in netinfo:
        if 'edge-cloud-rtt' in netinfo['spec']:
            status['service-metrics']['network']['edge-cloud-rtt-ms']['value'] = time_to_ms_converter(netinfo['spec']['edge-cloud-rtt'])
            status['service-metrics']['network']['edge-cloud-rtt-ms']['last-update'] = time.time()
        if 'cloud-edge-bps' in netinfo['spec']:
            status['service-metrics']['network']['cloud-edge-bps']['value'] = bitrate_to_bps_converter(netinfo['spec']['cloud-edge-bps'])
            status['service-metrics']['network']['cloud-edge-bps']['last-update'] = time.time()
        if 'edge-cloud-bps' in netinfo['spec']:
            status['service-metrics']['network']['edge-cloud-bps']['value'] = bitrate_to_bps_converter(netinfo['spec']['edge-cloud-bps'])
            status['service-metrics']['network']['edge-cloud-bps']['last-update'] = time.time()
    return

def parse_yaml():
    global gma_config, status

    logger.info(f"Parse yaml files")

    # compute the pod/deployment regex for each service

    for area in areas:
        for sc in gma_config['spec']['app']['services']:
            # compute the pod regex for the edge area
            if area == 'edge-area':
                items = sc['instances']['edge-yamls']
            else:
                items = sc['instances']['cloud-yamls']
            s = status['service-info'][sc['name']]
            for item in items:
                yaml_to_apply = item
                with open(yaml_to_apply) as f:
                    complete_yaml = yaml.load_all(f,Loader=yaml.FullLoader)
                    for partial_yaml in complete_yaml:
                        if partial_yaml['kind'] == 'Deployment' or partial_yaml['kind'] == 'StatefulSet' or partial_yaml['kind'] == 'DaemonSet':
                            # update pod information
                            if s['regex'][area]['pod'] == '':
                                s['regex'][area]['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                            else:
                                s['regex'][area]['pod'] = f'{s['regex'][area]['pod']}|{partial_yaml['metadata']['name']}-.*'
                            
                            if status['global-regex'][area]['pod'] == '':
                                status['global-regex'][area]['pod'] = f'{partial_yaml['metadata']['name']}-.*'
                            else:
                                status['global-regex'][area]['pod'] = f'{status['global-regex'][area]['pod']}|{partial_yaml['metadata']['name']}-.*'    
                            
                            # update workload information
                            if s['regex'][area]['workload']['regex'] != '':
                                logger.critical(f"Multiple deployments/statefulset/daemonset for the service {sc['name']} in the cloud-area not supported")
                                exit(1)
                            if partial_yaml['kind'] == 'Deployment':
                                s['regex'][area]['workload']['type'] == 'deployment'
                                s['regex'][area]['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                            if partial_yaml['kind'] == 'StatefulSet':
                                s['regex'][area]['workload']['type'] == 'statefulset'
                                s['regex'][area]['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                            if partial_yaml['kind'] == 'DaemonSet':
                                s['regex'][area]['workload']['type'] == 'daemonset'
                                s['regex'][area]['workload']['regex'] = f'{partial_yaml['metadata']['name']}'
                                                                                                                                    
                            if status['global-regex'][area]['workload'] == '':
                                status['global-regex'][area]['workload'] = f'{partial_yaml['metadata']['name']}'
                            else:
                                status['global-regex'][area]['workload'] = f'{status['global-regex'][area]['workload']}|{partial_yaml['metadata']['name']}'
                            for container in partial_yaml['spec']['template']['spec']['containers']:
                                if 'resources' in container and 'requests' in container['resources'] and 'cpu' in container['resources']['requests']:
                                    status['service-metrics']['qcpu'][area]['value'][s['id']] = status['service-metrics']['qcpu'][area]['value'][s['id']] + cpu_to_sec(container['resources']['requests']['cpu'])
                                if 'resources' in container and 'requests' in container['resources'] and 'memory' in container['resources']['requests']:
                                    status['service-metrics']['qmem'][area]['value'][s['id']] = status['service-metrics']['qmem'][area]['value'][s['id']] + mem_to_byte(container['resources']['requests']['memory'])
                        # update hpa information
                        if partial_yaml['kind'] == 'HorizontalPodAutoscaler' :
                            if s['regex'][area]['hpa'] == '':
                                s['regex'][area]['hpa'] = f'{partial_yaml['metadata']['name']}'
                                status['service-metrics']['hpa'][area]['min-replicas'][s['id']] = int(partial_yaml['spec']['minReplicas'])
                                status['service-metrics']['hpa'][area]['max-replicas'][s['id']] = int(partial_yaml['spec']['maxReplicas'])
                                try:
                                    status['service-metrics']['hpa'][area]['cpu-threshold'][s['id']] = float(partial_yaml['spec']['metrics'][0]['resource']['target']['averageUtilization'])/100.0
                                except KeyError:
                                    status['service-metrics']['hpa'][area]['cpu-threshold'][s['id']] = 0.6
                                    logger.warning(f"No HPA cpu-threshold for the service {sc['name']} in the {area} area, using default value 0.6")
                            else:
                                logger.critical(f"Multiple HPA for the service {sc['name']} in the {area} not supported")
                                exit(1)
                            if status['global-regex'][area]['hpa'] == '':
                                status['global-regex'][area]['hpa'] = f'{partial_yaml['metadata']['name']}'
                            else:
                                status['global-regex'][area]['hpa'] = f'{status['global-regex'][area]['hpa']}|{partial_yaml['metadata']['name']}'
        
        
def init():
    global gma_config, status, service_id_to_name

    logger.info(f"Init control dictionaries")

    status = dict() # Global Status dictionary
    status['service-info'] = dict() # Services status dictionary
    
    # Initialize the service information dictionary. It does not contain the metrics. The metrics are stored in the service_metrics dictionary
    mid = 0 # microservice id
    services=status['service-info']
    for s in gma_config['spec']['app']['services']:
            services[s['name']]=dict()
            # Initialize the service id
            if gma_config['spec']['app']['explicit-service-id']:
                services[s['name']]['id'] = s['id']
                if s['id'] > mid:
                    mid = s['id']+1 # needed for istio-ingress id
            else:
                services[s['name']]['id'] = mid
                mid = mid +1
            
            # Initialize the regex strings
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
    for service_name in status['service-info']:
        service_id_to_name[status['service-info'][service_name]['id']] = service_name
    

    status['global-regex'] = dict() # Global regex dictionary
    status['global-regex']['edge-area'] = dict()
    status['global-regex']['cloud-area'] = dict()
    status['global-regex']['edge-area']['pod'] = ''
    status['global-regex']['edge-area']['workload'] = ''
    status['global-regex']['edge-area']['hpa'] = ''
    status['global-regex']['cloud-area']['pod'] = ''
    status['global-regex']['cloud-area']['workload'] = ''
    status['global-regex']['cloud-area']['hpa'] = ''
    
    # Initialize service metrics

    status['service-metrics'] = dict()
    status['service-metrics']['n-services']= mid+1 # number of microservices
    M = status['service-metrics']['n-services']

    status['service-metrics']['fm'] = dict()
    status['service-metrics']['fm']['info'] = 'Call frequency matrix'
    status['service-metrics']['fm']['value'] = np.zeros((M,M),dtype=float)
    status['service-metrics']['fm']['last-update'] = 0 # last update time

    status['service-metrics']['response-length'] = dict()
    status['service-metrics']['response-length']['info'] = 'Response size vector in bytes'
    status['service-metrics']['response-length']['value'] = np.zeros(M,dtype=float)
    status['service-metrics']['response-length']['last-update'] = 0 # last update time

    status['service-metrics']['hpa'] = dict()
    status['service-metrics']['hpa']['cloud-area'] = dict()
    status['service-metrics']['hpa']['cloud-area']['info'] = 'hpa vectors for cloud area'
    status['service-metrics']['hpa']['cloud-area']['current-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['cloud-area']['desired-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['cloud-area']['old-current-replicas'] = np.zeros(M, dtype=int)
    status['service-metrics']['hpa']['cloud-area']['min-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['cloud-area']['max-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['cloud-area']['cpu-threshold'] = np.ones(M,dtype=int)*0.6
    status['service-metrics']['hpa']['cloud-area']['last-update'] = 0 # last update time
    status['service-metrics']['hpa']['edge-area'] = dict()
    status['service-metrics']['hpa']['edge-area']['info'] = 'Replicas vector for edge area'
    status['service-metrics']['hpa']['edge-area']['current-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['edge-area']['desired-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['edge-area']['old-current-replicas'] = np.zeros(M, dtype=int)
    status['service-metrics']['hpa']['edge-area']['min-replicas'] = np.zeros(M, dtype=int)
    status['service-metrics']['hpa']['edge-area']['max-replicas'] = np.zeros(M,dtype=int)
    status['service-metrics']['hpa']['edge-area']['cpu-threshold'] = np.ones(M,dtype=int)*0.6
    status['service-metrics']['hpa']['edge-area']['last-update'] = 0 # last update time
    
    status['service-metrics']['ucpu'] = dict()
    status['service-metrics']['ucpu']['cloud-area'] = dict()
    status['service-metrics']['ucpu']['cloud-area']['info'] = 'Actual CPU utilizatiion vector in seconds per second for cloud area'
    status['service-metrics']['ucpu']['cloud-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['ucpu']['cloud-area']['last-update'] = 0 # last update time
    status['service-metrics']['ucpu']['edge-area'] = dict()
    status['service-metrics']['ucpu']['edge-area']['info'] = 'Actual CPU utilizatiion vector in seconds per second for edge area'
    status['service-metrics']['ucpu']['edge-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['ucpu']['edge-area']['last-update'] = 0 # last update time
    
    status['service-metrics']['umem'] = dict()
    status['service-metrics']['umem']['cloud-area'] = dict()
    status['service-metrics']['umem']['cloud-area']['info'] = 'Actual memory utilizatiion vector in bytes for cloud area'
    status['service-metrics']['umem']['cloud-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['umem']['cloud-area']['last-update'] = 0 # last update time
    status['service-metrics']['umem']['edge-area'] = dict()
    status['service-metrics']['umem']['edge-area']['info'] = 'Actual memory utilizatiion vector in bytes for edge area'
    status['service-metrics']['umem']['edge-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['umem']['edge-area']['last-update'] = 0 # last update time
    
    status['service-metrics']['qcpu'] = dict()
    status['service-metrics']['qcpu']['cloud-area'] = dict()
    status['service-metrics']['qcpu']['cloud-area']['info'] = 'Requested CPU per pod in seconds per second for cloud area'
    status['service-metrics']['qcpu']['cloud-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['qcpu']['cloud-area']['last-update'] = 0 # last update time
    status['service-metrics']['qcpu']['edge-area'] = dict()
    status['service-metrics']['qcpu']['edge-area']['info'] = 'Requested CPU per pod in seconds per second for edge area'
    status['service-metrics']['qcpu']['edge-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['qcpu']['edge-area']['last-update'] = 0 # last update time
    
    status['service-metrics']['qmem'] = dict()
    status['service-metrics']['qmem']['cloud-area'] = dict()
    status['service-metrics']['qmem']['cloud-area']['info'] = 'Requested Mem per pod in seconds per second for cloud area'
    status['service-metrics']['qmem']['cloud-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['qmem']['cloud-area']['last-update'] = 0 # last update time
    status['service-metrics']['qmem']['edge-area'] = dict()
    status['service-metrics']['qmem']['edge-area']['info'] = 'Requested Mem per pod in seconds per second for edge area'
    status['service-metrics']['qmem']['edge-area']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['qmem']['edge-area']['last-update'] = 0 # last update time


    status['service-metrics']['service-lambda'] = dict()   
    status['service-metrics']['service-lambda']['info'] = 'Request rate vector in req/s'
    status['service-metrics']['service-lambda']['value'] = np.zeros(M, dtype=float)
    status['service-metrics']['service-lambda']['last-update'] = 0 # last update time
    
    status['service-metrics']['edge-user-delay']=dict()
    status['service-metrics']['edge-user-delay']['value'] = 0.0 # last update time
    status['service-metrics']['edge-user-delay']['info'] = 'Average edge user delay in ms' # last update time
    status['service-metrics']['edge-user-delay']['last-update'] = 0 # last update time

    status['service-metrics']['edge-user-delay-quantile']=dict()
    status['service-metrics']['edge-user-delay-quantile']['value'] = 0.0
    status['service-metrics']['edge-user-delay-quantile']['info'] = 'Edge user delay quantile in ms'
    status['service-metrics']['edge-user-delay-quantile']['last-update'] = 0 # last update time

    status['service-metrics']['edge-user-target-delay']=dict()
    status['service-metrics']['edge-user-target-delay']['value'] = 0.0 # last update time
    status['service-metrics']['edge-user-target-delay']['info'] = 'Average edge user target delay in ms' # last update time
    status['service-metrics']['edge-user-target-delay']['last-update'] = 0 # last update time
    
    status['service-metrics']['network'] = dict()
    status['service-metrics']['network']['edge-cloud-rtt-ms'] = dict()
    status['service-metrics']['network']['edge-cloud-rtt-ms']['value'] = time_to_ms_converter(gma_config['spec']['network']['edge-cloud-rtt-ms'])
    status['service-metrics']['network']['edge-cloud-rtt-ms']['info'] = 'Round trip time from edge area to cloud area in ms'
    status['service-metrics']['network']['edge-cloud-rtt-ms']['last-update'] = 0 # last update time
    status['service-metrics']['network']['edge-cloud-rtt-multiplier'] = dict()
    status['service-metrics']['network']['edge-cloud-rtt-multiplier']['value'] = gma_config['spec']['network']['edge-cloud-rtt-multiplier']
    status['service-metrics']['network']['edge-cloud-rtt-multiplier']['info'] = 'The RTT multiplier is applied to network RTT to obtain gRPC/HTTP-level round-trip time. Depends on the application. Configure with offline measurements'
    status['service-metrics']['network']['edge-cloud-rtt-multiplier']['last-update'] = 0 # last update time
    status['service-metrics']['network']['cloud-edge-bps'] = dict()
    status['service-metrics']['network']['cloud-edge-bps']['value'] = bitrate_to_bps_converter(gma_config['spec']['network']['cloud-edge-bps'])
    status['service-metrics']['network']['cloud-edge-bps']['info'] = 'Network capacity in bit per second from cloud area to edge area in bps'
    status['service-metrics']['network']['cloud-edge-bps']['last-update'] = 0 # last update time
    status['service-metrics']['network']['edge-cloud-bps'] = dict()
    status['service-metrics']['network']['edge-cloud-bps']['value'] = bitrate_to_bps_converter(gma_config['spec']['network']['edge-cloud-bps'])
    status['service-metrics']['network']['edge-cloud-bps']['info'] = 'Network capacity in bit per second from edge area to cloud area in bps'
    status['service-metrics']['network']['edge-cloud-bps']['last-update'] = 0 # last update time

    status['service-metrics']['cost'] = dict()
    status['service-metrics']['cost']['edge-area'] = dict()
    status['service-metrics']['cost']['edge-area']['cpu'] = dict()
    status['service-metrics']['cost']['edge-area']['cpu']['value'] = gma_config['spec']['edge-area']['cost']['cpu']
    status['service-metrics']['cost']['edge-area']['cpu']['info'] = 'Cost of CPU in the edge area per hour'
    status['service-metrics']['cost']['edge-area']['memory'] = dict()
    status['service-metrics']['cost']['edge-area']['memory']['value'] = gma_config['spec']['edge-area']['cost']['memory']
    status['service-metrics']['cost']['edge-area']['memory']['info'] = 'Cost of memory in the edge area per GB'
    status['service-metrics']['cost']['edge-area']['network'] = dict()
    status['service-metrics']['cost']['edge-area']['network']['value'] = gma_config['spec']['edge-area']['cost']['memory']
    status['service-metrics']['cost']['edge-area']['network']['info'] = 'Cost of external network traffic for the edge area per GB'

    status['service-metrics']['cost']['cloud-area'] = dict()
    status['service-metrics']['cost']['cloud-area']['cpu'] = dict()
    status['service-metrics']['cost']['cloud-area']['cpu']['value'] = gma_config['spec']['cloud-area']['cost']['cpu']
    status['service-metrics']['cost']['cloud-area']['cpu']['info'] = 'Cost of CPU in the cloud area per hour'
    status['service-metrics']['cost']['cloud-area']['memory'] = dict()
    status['service-metrics']['cost']['cloud-area']['memory']['value'] = gma_config['spec']['cloud-area']['cost']['memory']
    status['service-metrics']['cost']['cloud-area']['memory']['info'] = 'Cost of memory bytes in the cloud area per GB'
    status['service-metrics']['cost']['cloud-area']['network'] = dict()
    status['service-metrics']['cost']['cloud-area']['network']['value'] = gma_config['spec']['cloud-area']['cost']['memory']
    status['service-metrics']['cost']['cloud-area']['network']['info'] = 'Cost of external network traffic for the cloud area per GB'

    status['service-metrics']['me-resource-scaling'] = dict()
    status['service-metrics']['me-resource-scaling']['info'] = 'Cloud-to-edge multi-edge resource scaling factor'
    status['service-metrics']['me-resource-scaling']['value'] = np.ones(M, dtype=float) * gma_config['spec']['edge-area']['default-resource-scaling']
    status['service-metrics']['me-resource-scaling']['last-update'] = 0 # last update time

    # Get the pod/deployment regex for each service
    parse_yaml()
    check_inits()

def check_inits():
    # check workload exists for any service
    for service_name in status['service-info']:
        if status['service-info'][service_name]['regex']['cloud-area']['workload']['regex'] == '':
            logger.critical(f"Workload not found for service {service_name} in the cloud-area")
            exit(1)
        if status['service-info'][service_name]['regex']['edge-area']['workload']['regex'] == '':
            logger.critical(f"Workload not found for service {service_name} in the edge-area")
            exit(1)
    # check workloads for cloud area and edge have the same type
    for service_name in status['service-info']:
        if status['service-info'][service_name]['regex']['cloud-area']['workload']['type'] != status['service-info'][service_name]['regex']['edge-area']['workload']['type']:
            logger.critical(f"Workload type for service {service_name} in the cloud-area and edge-area are different")
            exit(1)
    # check hpa exists for any service
    for service_name in status['service-info']:
        if status['service-info'][service_name]['regex']['cloud-area']['hpa'] == '':
            logger.critical(f"HPA not found for service {service_name} in the cloud-area")
            exit(1)
        if status['service-info'][service_name]['regex']['edge-area']['hpa'] == '':
            logger.critical(f"HPA not found for service {service_name} in the edge-area")
            exit(1)
    #TODO check the node of the cluster have different topology labels

    

def cpu_to_sec(cpu_string):
    cpu_string = str(cpu_string)
    if cpu_string.endswith("m"):
        value = float(cpu_string.split("m")[0])/1000.0
    else:
        value = float(cpu_string)
    return value

def mem_to_byte(mem_string):
    mem_string = str(mem_string)
    if mem_string.endswith("m"):
        value = float(mem_string.split("M")[0])*1e6
    else:
        value = float(mem_string)
    return value

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
    # hpa_running: Camping with HPA is runnning
    # Camping: periodic monitoring of the system and no need to take action. No HPA running.
    # Offload_alarm: offload delay threshold is reached; check if this state persist for a while
    # Unoffload_alarm: unoffload delay threshold is reached; check if this state persist for a while
    # Offloading: offload action in progress
    # Unoffloading: unoffload action in progress

    def __init__(self):
        self.run()
        return

    def hpa_running(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Camping State (HPA Running)')
        logger.info(f'sleeping for {stabilizaiton_window_sec} stabilization sec')
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

        update_ingress_delay()
        update_ingress_delay_quantile()

        # check average delay violation for offloading
        logger.info(f'user delay: {status['service-metrics']['edge-user-delay']['value']} ms')
        logger.info(f'user delay quantile {delay_quantile}: {status['service-metrics']['edge-user-delay-quantile']['value']} ms')
        
        if status['service-metrics']['edge-user-delay']['value'] > offload_delay_threshold_ms:
            if np.all(status['service-metrics']['hpa']['edge-area']['current-replicas'][:-1] > 0):
                logger.warning('All microservice in the edge area, can not offload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sync sec')
                time.sleep(sync_period_sec)
                return     
            else:
                self.next = self.offload_alarm
                return
        
        # check quantile delay violation for offloading
        if status['service-metrics']['edge-user-delay-quantile']['value'] > offload_delay_quantile_threshold_ms:
            logger.info('Delay below offload quantile threshold')
            if np.all(status['service-metrics']['hpa']['edge-area']['current-replicas'][:-1] > 0):
                logger.warning('All microservice in the edge area, can not offload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sync sec')
                time.sleep(sync_period_sec)
                return     
            else:
                self.next = self.offload_alarm
                return
        
        unoffload_cond1 = False  # for unoffloading both avg and quantile condition must be satisfied
        unoffload_cond2 = False 
        # check average delay violation for unoffloading
        if status['service-metrics']['edge-user-delay']['value'] < unoffload_delay_threshold_ms:
            logger.info('Delay below unoffload threshold')
            if np.all(status['service-metrics']['hpa']['edge-area']['current-replicas'][:-1] == 0):
                logger.warning('No microservice in the edge area, can not unoffload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sync sec')
                time.sleep(sync_period_sec)
                return 
            else:
                unoffload_cond1 = True

        # check delay quantile violation for unoffloading
        if status['service-metrics']['edge-user-delay-quantile']['value'] < unoffload_delay_quantile_threshold_ms:
            logger.info('Delay below unoffload quantile threshold')
            if np.all(status['service-metrics']['hpa']['edge-area']['current-replicas'][:-1] == 0):
                logger.warning('No microservice in the edge area, can not unoffload more')
                self.next = self.camping
                logger.info(f'sleeping for {sync_period_sec} sync sec')
                time.sleep(sync_period_sec)
                return
            else:
                unoffload_cond2 = True

        if unoffload_cond1 and unoffload_cond2:
            self.next = self.unoffload_alarm
            return
        else:
            self.next = self.camping
            logger.info(f'sleeping for {sync_period_sec} sync sec')
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
            update_ingress_delay()
            update_ingress_delay_quantile()
            logger.info(f'user delay: {status['service-metrics']['edge-user-delay']['value']} ms')
            logger.info(f'user delay quantile {delay_quantile}: {status['service-metrics']['edge-user-delay-quantile']['value']} ms')
            if status['service-metrics']['edge-user-delay']['value'] > offload_delay_threshold_ms or status['service-metrics']['edge-user-delay-quantile']['value'] > offload_delay_quantile_threshold_ms:
                logger.info(f'sleeping for {stabilizaiton_window_sec-i*stabilization_cycle_sec} stabilization sec')
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
            update_ingress_delay()
            update_ingress_delay_quantile()
            logger.info(f'user delay: {status['service-metrics']['edge-user-delay']['value']} ms')
            logger.info(f'user delay quantile {delay_quantile}: {status['service-metrics']['edge-user-delay-quantile']['value']} ms')
            if status['service-metrics']['edge-user-delay']['value'] < unoffload_delay_threshold_ms and status['service-metrics']['edge-user-delay-quantile']['value'] < unoffload_delay_quantile_threshold_ms:
                logger.info(f'sleeping for {stabilizaiton_window_sec-i*stabilization_cycle_sec} stabilization sec')
                time.sleep(stabilization_cycle_sec)
            else:
                self.next = self.camping
                return
        self.next = self.unoffloading
        return
    
    def offloading(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Offloading')
        update_full_metrics()
        logger.info(f'user delay: {status['service-metrics']['edge-user-delay']['value']} ms')
        logger.info(f'user delay quantile {delay_quantile}: {status['service-metrics']['edge-user-delay-quantile']['value']} ms')
            
        offload_type = '' # quantile-driven or avg-driven

        if status['service-metrics']['edge-user-delay']['value'] > offload_delay_threshold_ms:
            offload_type = 'avg-driven'
            logger.info('avg-driven offloading')
        else:
            offload_type = 'quantile-driven'
            logger.info('quantile-driven offloading')
        
        offload_parameters = status['service-metrics'].copy()
        offload_parameters['ucpu']['cloud-area']['value'] = np.multiply(status['service-metrics']['me-resource-scaling']['value'],offload_parameters['ucpu']['cloud-area']['value']) # scaling the cloud cpu resources used by requests from the edge area
        offload_parameters['umem']['cloud-area']['value'] = np.multiply(status['service-metrics']['me-resource-scaling']['value'],offload_parameters['umem']['cloud-area']['value']) # scaling the cloud memory resources used by requests from the edge area
        
        if offload_type == 'avg-driven':
            target_delay_ms = unoffload_delay_threshold_ms + (offload_delay_threshold_ms-unoffload_delay_threshold_ms)/2.0
        else:
            target_delay_ms = offload_parameters['edge-user-delay']['value'] * delay_quantile_multiplier
        
        target_delay_ms = max(target_delay_ms, status['service-metrics']['edge-user-delay']['value']-max_delay_reduction_ms)
        offload_parameters['edge-user-target-delay']['value'] = target_delay_ms
        
        offload_parameters['edge-user-target-delay']['last-update'] = time.time()
        
        offload_parameters['optimizer'] = gma_config['spec']['optimizer'].copy()

        logger.info(f"Offloading with target delay reduction {offload_parameters['edge-user-delay']['value']-offload_parameters['edge-user-target-delay']['value']} ms ")
        # offloading logic
        result_list = Strategy_Connector.Compute_Placement(offload_parameters,action='offloading')
        logger.info(f"{result_list[1]['info']}")
        apply_configuration(result_list)
        logger.info(f'sleeping for {stabilizaiton_window_sec} stabilization sec')
        time.sleep(stabilizaiton_window_sec)
        # offloading done
        self.next = self.camping
        return
    
    def unoffloading(self):
        logger.info('_________________________________________________________')
        logger.info('Entering Unoffloading')
        update_full_metrics()

        unoffload_type = '' # quantile-driven or avg-driven

        if status['service-metrics']['edge-user-delay']['value'] < unoffload_delay_threshold_ms:
            unoffload_type = 'avg-driven'
            logger.info('avg-driven unoffloading')
        else:
            unoffload_type = 'quantile-driven'
            logger.info('quantile-driven unoffloading')

        unoffload_parameters = status['service-metrics'].copy()
        unoffload_parameters['ucpu']['cloud-area']['value'] = np.multiply(status['service-metrics']['me-resource-scaling']['value'],unoffload_parameters['ucpu']['cloud-area']['value']) # scaling the cloud cpu resources used by requests from the edge area
        unoffload_parameters['umem']['cloud-area']['value'] = np.multiply(status['service-metrics']['me-resource-scaling']['value'],unoffload_parameters['umem']['cloud-area']['value']) # scaling the cloud memory resources used by requests from the edge area
        if unoffload_type == 'avg-driven':
            target_delay_ms = unoffload_delay_threshold_ms + (offload_delay_threshold_ms-unoffload_delay_threshold_ms)/2.0
        else:
            target_delay_ms = unoffload_parameters['edge-user-delay']['value'] / delay_quantile_multiplier
        
        target_delay_ms = min(target_delay_ms, status['service-metrics']['edge-user-delay']['value']+max_delay_increase_ms)
        unoffload_parameters['edge-user-target-delay']['value'] = target_delay_ms

        unoffload_parameters['edge-user-target-delay']['last-update'] = time.time()
        
        unoffload_parameters['optimizer'] = gma_config['spec']['optimizer'].copy()
        
        logger.info(f"Unoffloading with target delay increase {unoffload_parameters['edge-user-target-delay']['value']-unoffload_parameters['edge-user-delay']['value']}ms ")
        # unoffloading logic
        result_list = Strategy_Connector.Compute_Placement(unoffload_parameters,action='unoffloading')
        logger.info(f"{result_list[1]['info']}")
        apply_configuration(result_list)
        logger.info(f'sleeping for {stabilizaiton_window_sec} stabilization sec')
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
    config_env=environ.get('GMA_CONFIG', './GMAConfig.yaml')
    log_env=environ.get('GMA_LOG_LEVEL', 'INFO')

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
        logger.critical(f"Config file not found: {args.configfile}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        logger.critical(f"Error in configuration file: {exc}")
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
        prom_client = PrometheusConnect(url=gma_config['spec']['telemetry']['prometheus-url'], disable_ssl=True)
    except PrometheusApiClientException:
        logger.critical(f"Error connecting to Prometheus server: {gma_config['spec']['prometheus-url']}")
        sys.exit(1)

    # LOad the optimizer connector
    try:
        Strategy_Connector = importlib.import_module(name=gma_config['spec']['optimizer']['connector'], package='strategies')
    except Exception as e:
        logger.critical(f"Error loading the optimizer Compute_Placement: {gma_config['spec']['optimizer']['connector']}")
        sys.exit(1)

    # Initialize the control dictionary
    areas = ['edge-area','cloud-area']  # set of areas, sequence matters for hpa check
    init()

    # global variables short-names
    # telemetry
    sync_period_sec = time_to_ms_converter(gma_config['spec']['telemetry']['sync-period'])/1000
    query_period_str = gma_config['spec']['telemetry']['query-period']
    query_period_sec = time_to_ms_converter(query_period_str)/1000
    stabilizaiton_window_sec = time_to_ms_converter(gma_config['spec']['telemetry']['stabilization-window'])/1000
    
    #slo
    offload_delay_threshold_ms = time_to_ms_converter(gma_config['spec']['slo']['offload-delay-threshold'])
    unoffload_delay_threshold_ms = time_to_ms_converter(gma_config['spec']['slo']['unoffload-delay-threshold'])
    offload_delay_quantile_threshold_ms =  time_to_ms_converter(gma_config['spec']['slo']['offload-delay-quantile-threshold'])
    unoffload_delay_quantile_threshold_ms =  time_to_ms_converter(gma_config['spec']['slo']['unoffload-delay-quantile-threshold'])
    delay_quantile = float(gma_config['spec']['slo']['delay-quantile'])
    default_resource_scaling = float(gma_config['spec']['edge-area']['default-resource-scaling'])
    
    #app
    namespace = gma_config['spec']['app']['namespace'] 
    M =  status['service-metrics']['n-services']  

    # cloud area
    cloud_istio_ingress_app = gma_config['spec']['cloud-area']['istio']['istio-ingress-source-app']
    cloud_istio_ingress_namespace = gma_config['spec']['cloud-area']['istio']['istio-ingress-namespace']
    cloud_pod_cidr_regex = gma_config['spec']['cloud-area']['pod-cidr-regex']
    
    # edge area
    edge_istio_ingress_app = gma_config['spec']['edge-area']['istio']['istio-ingress-source-app']
    edge_istio_ingress_namespace = gma_config['spec']['edge-area']['istio']['istio-ingress-namespace']
    edge_pod_cidr_regex = gma_config['spec']['edge-area']['pod-cidr-regex']

    # optimizer
    max_delay_reduction_ms = time_to_ms_converter(gma_config['spec']['optimizer']['max-delay-reduction'])
    max_delay_increase_ms = time_to_ms_converter(gma_config['spec']['optimizer']['max-delay-increase'])
    delay_quantile_multiplier = float(gma_config['spec']['optimizer']['delay-quantile-multiplier'])
    
    cluster=dict()
    for area in areas:
        cluster[area] = gma_config['spec'][area]['cluster']
    
    # Run the state machine
    sm = GMAStataMachine()