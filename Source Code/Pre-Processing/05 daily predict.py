# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:15:29 2019

@author: LEE SEONGGU
"""

import numpy as np

ori = np.load('china_processed.npy')
aver = np.zeros_like(ori)

# moving average
for time in range (23,ori.shape[0]) :
        for t_step in range (time-23,time+1) :            
            aver[time] += ori[t_step]
        aver[time] /= 24
    
timestep = 6
time_dimention = int(ori.shape[0]/timestep)

scaled_aver = np.zeros((time_dimention,ori.shape[1]))#,ori.shape[2],ori.shape[3]))

for scaled_time in range (0,time_dimention) :
    scaled_aver[scaled_time] = aver[scaled_time*6]
        
test1 = aver[:,0]       
refer = ori[:,0]
refer_scaled = scaled_aver[:,0]

np.save('china_processed_scaled.npy',scaled_aver )