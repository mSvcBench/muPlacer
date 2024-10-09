import numpy as np
def dp_builder_trace(node,trace, Fcm, M, S_b):
    children = np.argwhere(Fcm[node,0:M]>0).flatten()
    for child in children:
        if np.random.random() < Fcm[node,child]:
            trace[child] = 1
            trace = dp_builder_trace(child,trace,Fcm,S_b)
    return trace
