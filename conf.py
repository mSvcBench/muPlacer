PROMETHEUS_URL = "http://160.80.223.198:30000"  # Prometheus ip and port
CTX_CLUSTER2 = "kubernetes-admin1@cluster1.local"   # Kubernetes context of the edge cluster
CTX_CLUSTER2 = "kubernetes-admin@cluster.local"   # Kubernetes context of the cloud cluster
SLO = 160    # Service Level Objective (SLO) in ms
NAMESPACE = "edge"  # Kubernetes namespace of the application
SAVE_RESULTS = 0    # Save results in csv file (0 = don't save, 1 = save)
FOLDER = "temporary"    # Folder where to save the results
PLACEMENT_TYPE = "OE_PAMP"  # Placement type (OE_PAMP, IA, RANDOM, MFU)
MICROSERVICE_DIRECTORY = "/home/alex/Downloads/automate"  # Directory where microservice istance-set yaml are located
HPA_DIRECTORY = "/home/alex/Downloads/hpa"  # Directory where HPA yaml are located
NE = 1e9 # Edge-cloud bit rate
