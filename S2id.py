def S2id(S):
    # convert state value to decimal state id
    S = [int(x) for x in S]
    num_bin = ''.join(map(str, S))
    Sid = int(num_bin, 2) + 1
    return Sid