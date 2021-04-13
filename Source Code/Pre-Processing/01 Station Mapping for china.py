
"""
Created on Mon Sep 10 18:18:12 2018

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

def grid(v1, v2) :   # Latitude, Longitude
    RE = 6371.00877 # 지구 반경(km) 
    GRID = 5.0 # 격자 간격(km) 
    SLAT1 = 30.0 # 투영 위도1(degree) 
    SLAT2 = 60.0 # 투영 위도2(degree) 
    OLON = 126.0 # 기준점 경도(degree) 
    OLAT = 38.0 # 기준점 위도(degree) 
    XO = 43 # 기준점 X좌표(GRID) 
    YO = 136 # 기1준점 Y좌표(GRID) 
    DEGRAD = math.pi / 180.0 
    RADDEG = 180.0 / math.pi 
    re = RE / GRID; 
    slat1 = SLAT1 * DEGRAD 
    slat2 = SLAT2 * DEGRAD 
    olon = OLON * DEGRAD 
    olat = OLAT * DEGRAD 
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5) 
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn) 
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) 
    sf = math.pow(sf, sn) * math.cos(slat1) / sn 
    ro = math.tan(math.pi * 0.25 + olat * 0.5) 
    ro = re * sf / math.pow(ro, sn); 
    rs = {}; ra = math.tan(math.pi * 0.25 + (v1) * DEGRAD * 0.5) 
    ra = re * sf / math.pow(ra, sn) 
    theta = v2 * DEGRAD - olon 
    if theta > math.pi : 
            theta -= 2.0 * math.pi 
    if theta < -math.pi : 
            theta += 2.0 * math.pi 
    theta *= sn 
    rs['x'] = ra * math.sin(theta) + XO + 0.5
    rs['y'] = ro - ra * math.cos(theta) + YO + 0.5
    return rs['x'] , rs['y']



"""
1. cell당 하나의 station만 남도록 추리고, 추린 station은 mapping함.
"""

MAPPING = True
if(MAPPING) :
    dataset = read_csv('./data/미세먼지 측정소 주소_제주도.csv', header=0, index_col = 0, engine='python') 
#    Longitude =  dataset['Longitude'].values
#    Latitude =  dataset['Latitude'].values
#    print(len(Latitude))
#    x = np.zeros(len(Latitude))
#    y = np.zeros(len(Latitude))
#    for i in range(len(Latitude)) :
#        x[i], y[i] =  grid(Latitude[i], Longitude[i])
#        
    x_min =  int(dataset['X'].min())  #그리드의 최대최소값을 구함.
    x_max =  int(dataset['X'].max())
    y_min =  int(dataset['Y'].min())
    y_max =  int(dataset['Y'].max())
    
    stations_map = np.zeros((x_max-x_min+1, y_max-y_min+1))  # station index가 행렬에 맵핑된다
    using_stations = np.zeros(206) # 사용한 스테이션의 개수를 저장.
    use_station_num = 0
    for staion_index in range (dataset.shape[0]) :   # 그리드 매칭
                x_in = dataset.iloc[staion_index,3]-x_min
                y_in = dataset.iloc[staion_index,4]-y_min
                if stations_map[x_in][y_in] == 0 :   
                    stations_map[x_in][y_in] = dataset.iloc[staion_index,5]
                    using_stations[use_station_num]= staion_index
                    use_station_num += 1

    # drop하기 (particulate stations used로 중복제거하여 변환)
                    
"""
2. 측정소 코드 따오기
"""           

error = []

CODE_EXTRACTION = False
if(CODE_EXTRACTION) :
    particledataset = read_csv('./data/2014년 1분기.csv', header=0, index_col = 10, engine='python')  # 측정소 코드의 소스
    stationinfodataset = read_csv('./data/미세먼지 측정소 주소_Final.csv', header=0, index_col = 0, engine='python') 
    station_id = list() 
    for staion_index in range (stationinfodataset.shape[0]) : 
        try:
            name = stationinfodataset.iloc[staion_index,0]                 
            station_id.append(particledataset.loc[name,"측정소코드"].head(1)[0])
            error.append(particledataset.loc[name,"측정소코드"].head(1)[0])         
        except : error.append(name)
else :  
    station_id = dataset['측정소코드'].values
"""
3. 측정소 코드에 맞추어 데이터 변환
"""    
DATA_TRANS1 = True   
feature = ['측정소코드','측정일시','PM10']

if(DATA_TRANS1) :   
    file_num = 0
    for y in range(2014,2019):
        for qu in [1,2,3,4]:
            print(qu)
            file_num = file_num +1
            if(file_num==1):
                dataset =read_csv('./data/%d년 %d분기.csv' %(y,qu), header=0, index_col = 0, usecols = feature,  engine='python') 
                dataset = dataset.loc[station_id,:]
            else:
                tempDataset = read_csv('./data/%d년 %d분기.csv' %(y,qu), header=0, index_col = 0, usecols = feature, engine='python')
                tempDataset = tempDataset.loc[station_id,:]
                dataset = concat([dataset,tempDataset])  
    dataset.index.names = ['station_id']
    dataset = dataset.rename(columns={'측정일시':'date'})            
    dataset.to_csv('./output/2014-2018 all_PM10_제주도.csv',header=True, index=True)
#output은 측정소코드별 feature값을 열로 가지는 행렬


"""
4. 먼지측정소 데이터를 날짜에 맞춰 병렬화
"""
    
extractFeature = ['PM10']

DATA_TRANS2 = True   

if(DATA_TRANS2) :
    dataset = read_csv('./output/2014-2018 all_PM10_제주도.csv', header=0, index_col = 1, engine='python') 
    for fea in range(len(extractFeature)): #feature별 생성
        for i in range(len(station_id)):
                print(i)
                if(i==0):
                    pmSet = dataset.loc[dataset['station_id']==station_id[i],extractFeature[fea]]
                else:
                        pmSet = concat([pmSet, dataset.loc[dataset['station_id']==station_id[i], extractFeature[fea]]], axis = 1)
        pmSet.columns = station_id
        pmSet = pmSet.drop(pmSet.index[len(pmSet.index)-1],0) # weather와 행 개수 맞추기 위해 마지막 행 제거
        pmSet.to_csv('./output/2014-2018 all_제주도'+extractFeature[fea]+'.csv',header=True, index=True)
           
