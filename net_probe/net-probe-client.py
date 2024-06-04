from multiprocessing import Process
import time
from flask import Flask, jsonify, request
import numpy as np
import random
import string

import requests

# insert flask annotation here
app = Flask(__name__)

# @app.route('/start')
# def _start():
#     global measure_process_id, status, process_ids
#     if status == 'started':
#         return jsonify({'error': 'Already started'}), 400
#     server = request.args.get('server', default='net-probe-server', type=string) # server address
#     size = request.args.get('size', default=1e6, type=int) # size of the HTTP request
#     parallels = request.args.get('parallels', default=1, type=int) # number of parallel HTTP requests process\
#     process_ids = np.zeros(parallels)
#     for i in range(parallels):
#         measure_process = Process(target=measure, args=(server,))
#         measure_process.start()
#         process_ids[i] = measure_process.pid
#     status = 'started'
#     return jsonify({'status': 'started'}), 200

@app.route('/start')
def _start():
    global status
    if status == 'started':
        return jsonify({'error': 'Already started'}), 400
    server = request.args.get('server', default='net-probe-server', type=string) # server address
    size = request.args.get('size', default=1e6, type=int) # size of the HTTP request
    response = measure(server,s)
    if response.status_code != 200:
        print(f"HTTP GET failed with status code {response.status_code}")
    report={}
    report['response_time'] = response.elapsed.total_seconds()
    report['response_size'] = len(response.content)
    report['response_status'] = response.status_code
    report['response_bitrate'] = response.status_code
    return jsonify({'size': 'started'}), 200
    
    

    #parallels = request.args.get('parallels', default=1, type=int) # number of parallel HTTP requests process\
    # process_ids = np.zeros(parallels)
    # for i in range(parallels):
    #     measure_process = Process(target=measure, args=(server,))
    #     measure_process.start()
    #     process_ids[i] = measure_process.pid
    # status = 'started'
    return jsonify({'status': 'started'}), 200

def measure(server,s):
    global n_parallel, report
    s = requests.Session()
    response = s.get(f"http://{server}:5000/get?s={s}")
    return response

if __name__ == "__main__":
    # process id
    ongoing_processes = 0
    status = 'stopped'
    process_ids = None
    n_parallel = 1 # number of parallel HTTP processes
    external_background_traffic = 0 # external background traffic in Mbps
    
    app.run(host='0.0.0.0')