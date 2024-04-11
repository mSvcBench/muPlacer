from math import log2

# 

def id2S(Sid, Ns):
    S = list(bin(Sid - 1)[2:])
    S = list(map(int, S))
    S = [0] * (int(log2(Ns)) - len(S)) + S
    return S