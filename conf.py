PROMETHEUS_URL = "http://160.80.223.198:30000"  # Prometheus ip and port
# Contexts of the edge and cloud clusters are equal in case of Liqo
CTX_CLUSTER1 = "liqo-admin@kubernetes"   # Kubernetes context of the edge cluster
CTX_CLUSTER2 = "liqo-admin@kubernetes"   # Kubernetes context of the cloud cluster
SLO = 160    # Service Level Objective (SLO) in ms
NAMESPACE = "edge"  # Kubernetes namespace of the application
SAVE_RESULTS = 0    # Save results in csv file (0 = don't save, 1 = save)
FOLDER = "temporary"    # Folder where to save the results
PLACEMENT_TYPE = "OE_PAMP"  # Placement type (OE_PAMP, IA, RANDOM, MFU)
MICROSERVICE_DIRECTORY = "/Users/detti/muBench/SimulationWorkspace/affinity-yamls/no-region-specified"  # Directory where microservice instance-set yaml are located
HPA_DIRECTORY = "/Users/detti/muBench/SimulationWorkspace/affinity-yamls/hpa"  # Directory where HPA yaml are located
NE = 100e6 # Edge-cloud bit rate
