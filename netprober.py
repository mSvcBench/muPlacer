from flask import Flask, jsonify, request
import iperf3
import subprocess

app = Flask(__name__)

@app.route('/get', methods=['GET'])
def get_bandwidth_and_rtt():
    try:
        # Get parameters from request
        server_ip = request.args.get('server_ip', default="127.0.0.1", type=str)
        bandwidth_mbps = request.args.get('bandwidth_mbps', default=None, type=float)
        duration = request.args.get('duration', default=3, type=int)
        blksize = request.args.get('blksize', default=1340, type=int)
        # Create iperf3 client object
        client = iperf3.Client()
        client.server_hostname = server_ip
        client.port = 5201
        client.protocol = 'udp'
        client.duration = duration
        client.blksize = blksize
        client.num_streams = 1
        client.zerocopy = True
        client.json_output = True

        if bandwidth_mbps is not None:
            client.bandwidth = int(bandwidth_mbps * 1e6)  # Convert Mbps to bps

        # Run the iperf3 client
        result = client.run()
        if result.error:
            return jsonify({"error": result.error}), 500

        # Parse iperf3 results
        resultj = result.json
        measured_bandwidth_bps = resultj['intervals'][1]['streams'][0]['bits_per_second']
        measured_bandwidth_mbps = measured_bandwidth_bps / 1e6

        # Measure RTT using bash ping
        ping_result = subprocess.run(["ping", "-c", "4", server_ip], capture_output=True, text=True)
        if ping_result.returncode != 0:
            return jsonify({"error": "Ping failed", "details": ping_result.stderr}), 500

        # Parse RTT from ping output
        for line in ping_result.stdout.splitlines():
            if "rtt" in line or "round-trip" in line:
                rtt_stats = line.split("=")[1].split("/")  # Extract min/avg/max/mdev
                rtt_avg_ms = float(rtt_stats[1])  # Average RTT in ms
                break
        else:
            return jsonify({"error": "Unable to parse ping output"}), 500

        # Return results as JSON
        return jsonify({
            "edge-cloud-bps": measured_bandwidth_mbps,
            "edge-cloud-rtt": rtt_avg_ms
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)