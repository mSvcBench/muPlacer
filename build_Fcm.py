import numpy as np
from muPlacer import get_app_names

# Function that build call frequency matrix Fcm

def Fcm(prom, PERIOD):
    app_names = get_app_names() # Get app names with the relative function
    
    # add app name of istio ingress in app_names list for Fcm matrix
    app_istio_ingress = "istio-ingressgateway"
    app_names = app_names + [app_istio_ingress] # Add istio-ingress to app_names list

    # Create a matrix with numpy to store calling probabilities
    Fcm = np.zeros((len(app_names), len(app_names)), dtype=float)

    # Find and filter significant probabilities
    for i, src_app in enumerate(app_names):
        for j, dst_app in enumerate(app_names):
            # Check if source and destination apps are different
            if src_app != dst_app:
                # total requests that arrive to dst microservice from src microservice
                query1 = f'sum by (destination_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[{PERIOD}m]))'
                
                #total requests that arrive to the source microservice
                if src_app != "istio-ingressgateway":
                    # Query for istio-ingress
                    query2 = f'sum by (source_app) (rate(istio_requests_total{{reporter="destination", destination_app="{src_app}",response_code="200"}}[{PERIOD}m]))'
                else:
                    query2 = f'sum by (source_app) (rate(istio_requests_total{{source_app="{src_app}",reporter="destination", destination_app="{dst_app}",response_code="200"}}[{PERIOD}m]))' 

                #print(query1)
                #print(query2)
                r1 = prom.custom_query(query=query1)
                r2 = prom.custom_query(query=query2)

                # Initialize variables
                calling_probability = None
                v = 0
                s = 0

                # Extract values from queries
                if r1 and r2:
                    for result in r1:
                        if float(result["value"][1]) == 0: 
                            continue
                        v = result["value"][1]
                        #print("v =", v)
                    for result in r2:
                        if float(result["value"][1]) == 0:
                            continue
                        s = result["value"][1]
                        #print("s =", s)

                    # Calculate calling probabilities
                    if s == 0 or s == "Nan":
                        calling_probability = 0 # If there isn't traffic
                    else:
                        calling_probability = float(v) / float(s)
                        if calling_probability > 0.98:
                            calling_probability = 1
                        
                        calling_probability = round(calling_probability, 3)  # Round to 4 decimal places
                        Fcm[i, j] = calling_probability # Insert the value inside Fcm matrix              
    #print(Fcm)
    return Fcm