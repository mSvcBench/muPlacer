import random
import string
import numpy as np
import os
import time
import requests
from multiprocessing import Process
import iperf3
import subprocess


def iperf():
    # Code for the iperf thread goes here
    client = iperf3.Client()
    client.duration = 1
    client.server_hostname = server_ip
    client.port = 5201
    client.bandwidth = np.uint64(network_background_traffic * 1000000)
    client.protocol = 'udp'
    client.reverse = True
    client.duration = 3600
    result = client.run()


def http_get(s):
    response = session.get(f"http://{server_ip}:{server_port}/get?s={s}")
    if response.status_code != 200:
        print(f"HTTP GET failed with status code {response.status_code}")
    return response.text

if __name__ == "__main__":
    s_min = 1*1000 # minimum number of bytes to request
    s_max = 500*1000 # maximum number of bytes to request
    input_np_file = 'data.npy'
    out_np_file = 'data.npy'
    n_http_gets = 1000 # number of requests to make
    server_ip = '192.168.56.101'
    server_port = 5000
    user = 'ubuntu' # user name on server machine for ssh
    network_background_traffic_range = range(0, 45, 5)
    network_delay_range = range(0, 50, 10)
    network_bandwidth = 50 # network bandwidth in Mbits
    #network_background_traffic = 40 # network background traffic in Mbps
    #network_delay = 20   # network delay in ms

    for network_background_traffic in network_background_traffic_range:
        for network_delay in network_delay_range:
            # configure network bandwidth and delay
            subprocess.Popen(f"ssh {user}@{server_ip} 'sudo tc qdisc change dev eth1 root netem delay {network_delay}ms rate {network_bandwidth}Mbit'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

            session = requests.Session()

            if os.path.exists(input_np_file):
                np_data = np.load(input_np_file)
                # load the data
            else:
                np_data = np.empty((0, 5))

            # Create and start the iperf thread
            if network_background_traffic>0:
                iperf_process = Process(target=iperf)
                iperf_process.start()

            for i in range(n_http_gets):
                s = random.randint(s_min, s_max)
                start_time = time.time()
                http_get(s)
                end_time = time.time()
                execution_time = (end_time - start_time)*1000.0
                result = np.array([network_bandwidth, network_delay, network_background_traffic, s, execution_time],ndmin=2)
                np_data = np.append(np_data, result, axis=0)
                print(f"{i} - Time needed to get {s} bytes: {execution_time} ms, bk traffic {network_background_traffic}, net delay {network_delay}")
                time.sleep(0.2)
            np.save(out_np_file, np_data)
            session.close()
            iperf_process.terminate() if network_background_traffic>0 else 0