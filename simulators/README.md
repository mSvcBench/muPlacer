# Simulations
This folder contains simulations designed to evaluate different strategies and configurations for the system.

- **Cold-start Simulations**: These simulations assess system performance by restarting from a cloud-only configuration at the beginning of each test.

- **Warm-start Simulations**: These simulations evaluate system performance by modifying a configuration parameter and restarting from the previous state left by the strategy, rather than from a cloud-only configuration.



`sim_app_size.py`
Cold-start simulation varying the number of micorservices from 10 to 210

`sim_app_delay.py`
Cold-start simulation varying the delay from 60 to 180 ms

`sim_app_RTT.py`
Cold-start simulation varying the RTT from 40 to 200 ms

`sim_app_B.py`
Cold-start simulation varying the bandwidth from 0.2 to 1.8 Gbps

`sim_app_exp.py`
Cold-start simulation varying the expanding depth of SBMP from 1 to 5 and infinity

`sim_app_lambda_dny.py`
Warm-start simulation varying the lambda from 40 to 500 req/s and back
