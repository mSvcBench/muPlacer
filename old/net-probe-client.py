from pythonping import ping
import iperf3
import time

def ping_test(ping_server_ip, verbose=False):
    response = ping(ping_server_ip, verbose=verbose)
    return response.rtt_avg*1000

def iperf_test(iperf_server_ip, iperf_server_port, verbose=False):
    client = iperf3.Client()
    client.server_hostname = iperf_server_ip
    client.port = iperf_server_port
    client.duration = 120
    client.verbose = verbose
    client.reverse = True
    client.run()
    return client.sent_Mbps

