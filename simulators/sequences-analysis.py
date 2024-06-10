import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from os import environ
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS

import numpy as np
import matplotlib.pyplot as plt


sequence1= np.load('/home/ubuntu/Andrea/muPlacer/simulators/sequence.npy')
sequence2= np.load('/home/ubuntu/Andrea/muPlacer/simulators/sequence2.npy')
for i in range(len(sequence1)):
    if not np.array_equal(sequence1[:,i],sequence2[:,i]):
        print(f"sequence1: {np.argwhere(sequence1[:,i]>0).T[0]}\nsequence2: {np.argwhere(sequence2[:,i]>0).T[0]}")








