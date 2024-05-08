import numpy as np 

def computeDiTot(Nci, Di):

    # S : current state vector
    # Nci : number of instance call per user request of the current state
    # Fci : call frequency matrix of the current state
    # Di : internal delay of microservices
    
    return (np.sum(np.multiply(Nci,Di)))

def main():
    # Example usage
    Nci = [1, 1, 2]  # number of instance call per user request of the current state
    Di = [0.1, 0.2, 0.3]  # internal delay of microservices
    
    result = computeDiTot(Nci, Di)
    print("Total delay:", result)

if __name__ == "__main__":
    main()
