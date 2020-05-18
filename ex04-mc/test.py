# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:36:19 2020

@author: Gregor
"""


import numpy as np

states = np.zeros((2,10,10))
#states = np.reshape(states,(200,1))
rewards = []
for x in range(0,2):
    rewards.append([])
    for y in range(0,10):
        rewards[x].append([])
        for z in range(0,10): 
            rewards[x][y].append([])
            
            
            
x = [1,2]
print(x[True])
print(True)