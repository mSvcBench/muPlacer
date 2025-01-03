# create Call frequency matrix from mubench workmodel without databases
import json
import numpy as np

def createFcm(input_file):
    with open(input_file, "r", encoding="utf8") as file:
        workmodel = json.load(file)
    M = len(workmodel) + 1 # last one is the user
    Fcm = np.zeros((M,M))
    Rcpu = np.zeros(M)
    Rs = np.zeros(M)
    for ms in workmodel:
        Rcpu[name2id(ms)] = int(workmodel[ms]['cpu-limits'][:-1])/1000.0
        Rs[name2id(ms)] = int(workmodel[ms]['internal_service']['loader']['mean_bandwidth']*1000)
        for external_services_group in workmodel[ms]['external_services']:
            services = external_services_group['services']
            probabilities = external_services_group['probabilities']
            for dms in services:
                if dms in probabilities:
                    Fcm[name2id(ms)][name2id(dms)] = probabilities[dms]
                else:
                    Fcm[name2id(ms)][name2id(dms)] = 1
    Fcm[M-1][0] = 1 # user calls the first microservice
    return Fcm,Rcpu,Rs


def name2id(name):
    # first caracter of the service name is s, next ones are the id
    x = int(name[1:])
    return x
    

if __name__ == "__main__":
    Fcm,Rcpu,Rs=createFcm("simulators/workmodel.json")
    print(Fcm)
    print(Rcpu)
    print(Rs)