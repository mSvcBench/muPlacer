PROMETHEUS_URL = "http://160.80.223.198:30000"  # Prometheus ip and port
CTX_CLUSTER2 = "kubernetes-admin1@cluster1.local"   # Kubernetes context of the edge cluster
SLO = 70    # Service Level Objective (SLO) in ms
NAMESPACE = "edge"  # Namespace of the application
SAVE_RESULTS = 0    # Save results in csv file (0 = don't save, 1 = save)
POSITIONING_TYPE = "mfu"    # Type of positioning strategy used (autoplacer, only_cloud, random, mfu, IA)
FOLDER = "temporary"    # Folder where to save the results
PLACEMENT_TYPE = "OE_PAMP"  # Placement type (OE_PAMP, IA, RANDOM, MFU)
