# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:13:05 2019

@author: LEE SEONGGU
"""
import numpy as np



baekryongdo_timestep = 7 # -6~ 0
china_timestep = 37 # -36~ 0


added_feature = 37 + 7 

china_dataset = np.load('./output/china_processed.npy') 
baekryongdo_dataset = np.load('./output/combine_IDW_Baekryongdo.npy') 

combine_map_merged =  np.zeros((baekryongdo_dataset.shape[0], 
                                baekryongdo_dataset.shape[1],
                                baekryongdo_dataset.shape[2], 
                                baekryongdo_dataset.shape[3]+added_feature))

# 백령도 데이터 추가
start = baekryongdo_dataset.shape[3] 
combine_map_merged[:,:,:,:start] = baekryongdo_dataset
combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[0,0,0,0]

for time in range (1,baekryongdo_timestep):
    combine_map_merged[time,:,:,start+baekryongdo_timestep-time:start+baekryongdo_timestep] = baekryongdo_dataset[0:time,0,0,0]
    combine_map_merged[time,:,:,start:start+baekryongdo_timestep-time] = baekryongdo_dataset[0,0,0,0]


for time in range (baekryongdo_timestep, baekryongdo_dataset.shape[0]) :
    combine_map_merged[time,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[time-baekryongdo_timestep:time,0,0,0]

# 중국 데이터 추가
baijing_col = 1   

start = baekryongdo_dataset.shape[3]+baekryongdo_timestep
combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = china_dataset[0,baijing_col]

for time in range (1,china_timestep):
    combine_map_merged[time,:,:,start+china_timestep-time:start+china_timestep] = china_dataset[0:time,baijing_col]
    combine_map_merged[time,:,:,start:start+china_timestep-time] = china_dataset[0,baijing_col]


for time in range (china_timestep, china_dataset.shape[0]) :
    combine_map_merged[time,:,:,start:start+china_timestep] = china_dataset[time-china_timestep:time,baijing_col]

baekryongdo_dataset022 = baekryongdo_dataset[0,:,:,:]
baekryongdo_dataset0 = baekryongdo_dataset[:,0,0,:]
combine_map_merged0 =  combine_map_merged[:,0,0,:]
combine_map_mergedtest =  combine_map_merged[:,:,0,:]
np.save('./output/combine_IDW_china', combine_map_merged)
