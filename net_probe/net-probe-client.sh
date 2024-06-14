for i in {1..12}
do
   echo "Running iperf3 test: $i"
   iperf3 -c net-probe-server -p 5201 -t 10
done