# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:40:23 2019

@author: LEE SEONGGU
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math

china_dataset = read_csv('./data/china_1418.csv', header=0, index_col = 0,  engine='python') 
#china_dataset_ffill = china_dataset.fillna(method='ffill')
data_length = china_dataset.shape[0]
data_feature = china_dataset.shape[1]
china_processed = china_dataset.values


for feature in range(data_feature) :
    ff_index = 0 # 최신 vaild값
    bb_index = 0 # 다음시간에 대한 vaild값
    for time in range(data_length) :
        print(time)
        if(math.isnan(china_processed[time,feature])):
            #NAN 발견
             if(bb_index<=ff_index) : # bb index 갱신
                     bb_index = time+1
                     while(math.isnan(china_processed[bb_index,feature])) :
                         bb_index = bb_index + 1
             ff_vaild = china_processed[ff_index,feature] 
             bb_vaild = china_processed[bb_index,feature] 
             china_processed[time,feature] = ((time-ff_index)*bb_vaild + ff_vaild*(bb_index-time))/(bb_index-ff_index)
        else :
          ff_index = time # vaild값 갱신
 
          
          np.save('./output/china_processed', china_processed)