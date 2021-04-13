
from pandas import read_csv

import numpy as np

from math import sin, cos, sqrt, atan2, radians

from keras.regularizers import l1, l2, l1_l2

import scipy.sparse as sp
from keras.models import load_model

import matplotlib.pyplot as plt
import main_val as main

import math
import os 

import module_conv_develop_LSTM
import module_conv_LSTM
import module_conv_LSTM_sensor

#
def normalized(map_):
    remain_shape = 1
    for shape in range(map_.ndim-1) :
        remain_shape = remain_shape * np.size(map_,shape)
    map_n = map_.reshape(
            remain_shape,np.size(map_,-1))
    map_max_ = np.zeros(np.size(map_n,1))
    for i in range(np.size(map_max_)) :
        map_max_[i] = np.max(map_n[:,i])
        map_n[:,i] = map_n[:,i]/map_max_[i]
    return  map_max_

##normalized
#def normalized(map_):
#    map_n = map_.reshape(
#            np.size(map_,0)*np.size(map_,1)*np.size(map_,2),np.size(map_,3))
#    map_std_ = np.zeros(np.size(map_n,1))
#    for i in range(np.size(map_std_)) :
#        map_std_[i] = np.std(map_n[:,i])
#        map_n[:,i] = map_n[:,i]/map_std_[i]
#    return  map_std_
#     

def make_matrix(target_dataset, x_min, y_min) : 

    stations_xy_hash_table = np.zeros((target_dataset.shape[0],3))    

    for staion_index in range (target_dataset.shape[0]) :   # 그리드 매칭
                x_in = target_dataset.iloc[staion_index,3]-x_min
                y_in = target_dataset.iloc[staion_index,4]-y_min
                stations_xy_hash_table[staion_index][0] = 0 # 별의미 음없
                stations_xy_hash_table[staion_index][1] = int(x_in)
                stations_xy_hash_table[staion_index][2] = int(y_in)
    return stations_xy_hash_table, target_dataset.shape[0]

def get_matrix(matrix_name1, matrix_name2) : 
    d1 = read_csv(matrix_name1, header=0, index_col = 0, engine='python') 
    d2 = read_csv(matrix_name2, header=0, index_col = 0, engine='python') 
    x_min =  min([d1['X'].min(),d2['X'].min()])
    x_max =  max([d1['X'].max(),d2['X'].max()])
    y_min =  min([d1['Y'].min(),d2['Y'].min()])
    y_max =  max([d1['Y'].max(),d2['Y'].max()])
    x_size = x_max-x_min+1
    y_size = y_max-y_min+1  
    return d1, d2, x_size, y_size, x_min, y_min

def dataprocess( weather_feature, date_feature, stations_data_map, 
                 PM_station_num,weather_station_num,PM_xy_hash_table,weather_xy_hash_table):
    # ConvLSTM이외 다른 모델들을 위한 행렬-> 2D 데이터 프로세싱
        total_feature = PM_station_num*pollutant_feature+weather_station_num*weather_feature + date_feature
        poll_start = pollutant_feature*PM_station_num
        weather_start = poll_start + weather_feature*weather_station_num
        standard_set = np.zeros((stations_data_map.shape[0], total_feature))
        for ff in range(0,pollutant_feature) :  
            for station_num in range(PM_station_num) :
                x   = int(PM_xy_hash_table[station_num][1]) 
                y   = int(PM_xy_hash_table[station_num][2])          
                standard_set[:,PM_station_num*ff + station_num] =stations_data_map[:,x,y,ff]  
        for ff in range(weather_feature) :  
            for station_num in range(weather_station_num) : 
                x   = int(weather_xy_hash_table[station_num][1]) 
                y   = int(weather_xy_hash_table[station_num][2])   
                standard_set[:,poll_start + weather_station_num*ff + station_num] =stations_data_map[:,x,y,ff+pollutant_feature]  
        for ff in range(date_feature) : 
            standard_set[:,weather_start + ff] =stations_data_map[:,0,0,ff+pollutant_feature+weather_feature]  

        return standard_set
    
def dataprocess_in_averages( weather_feature, date_feature, stations_data_map, 
                 PM_station_num,weather_station_num,PM_xy_hash_table,weather_xy_hash_table):
    # ConvLSTM이외 다른 모델들을 위한 행렬-> 2D 데이터 프로세싱
    # PM / Poll / Weather / Date
    
        total_feature = PM_station_num+(pollutant_feature-1)+ weather_feature + date_feature
        pm_start = PM_station_num
        poll_start = pollutant_feature-1+PM_station_num
        weather_start = poll_start + weather_feature # weather는 평균냄
        standard_set = np.zeros((stations_data_map.shape[0], total_feature))
        
        for station_num in range(PM_station_num) :
            x   = int(PM_xy_hash_table[station_num][1]) 
            y   = int(PM_xy_hash_table[station_num][2])          
            standard_set[:,station_num] =stations_data_map[:,x,y,0]  
            
        for ff in range(0,pollutant_feature-1) :  
            for station_num in range(PM_station_num) :       
                x   = int(PM_xy_hash_table[station_num][1]) 
                y   = int(PM_xy_hash_table[station_num][2])          
                standard_set[:,pm_start + ff] +=stations_data_map[:,x,y,ff+1]  
            standard_set[:,pm_start + ff] /= PM_station_num    
            
        for ff in range(weather_feature) :  
            for station_num in range(weather_station_num) : 
                x   = int(weather_xy_hash_table[station_num][1]) 
                y   = int(weather_xy_hash_table[station_num][2])   
                standard_set[:,poll_start + ff] += stations_data_map[:,x,y,ff+pollutant_feature]  
            standard_set[:,poll_start + ff] /= weather_station_num    
                
        for ff in range(date_feature) : 
            standard_set[:,weather_start + ff] =stations_data_map[:,0,0,ff+pollutant_feature+weather_feature]  

        return standard_set
    
    
    
    
    #끝     
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
    return rs['x'], rs['y'] 

def mydistance(a1,b1,a2,b2): # a:longitude , b:latitude
    R = 6373.0
    x1 = radians(a1)
    y1 = radians(b1)
    x2 = radians(a2)
    y2 = radians(b2)
    dlon = x2 - x1
    dlat = y2 - y1
    a = sin(dlat / 2)**2 + cos(y1) * cos(y2) * sin(dlon / 2)**2 
    c = 2 * atan2(sqrt(a), sqrt(1 - a))    
    return R * c      

    

def graphprocess(standard_set, adj_set, target_dataset, stations_data_map, 
                 PM_station_num,PM_xy_hash_table):
    # ConvLSTM이외 다른 모델들을 위한 행렬-> 2D 데이터 프로세싱
        adj = np.zeros((PM_station_num,PM_station_num))   # 연결됨 안됨의 가중치
        adj_distance = np.zeros((PM_station_num,PM_station_num))  # 거리 가중치
        adj_x = np.zeros((PM_station_num,PM_station_num))  # x 좌표 의존적 가중치
        adj_y = np.zeros((PM_station_num,PM_station_num))  #y 좌표 의존적 가중치
 
        for i in range(PM_station_num) : 
            #데이터 행렬 만들기
            num = int(PM_xy_hash_table[i][0])
            x   = int(PM_xy_hash_table[i][1]) 
            y   = int(PM_xy_hash_table[i][2])
            for ff in range(0, stations_data_map.shape[3]) : 
                standard_set[:,i,ff] =stations_data_map[:,x,y,ff]  
            #인접행렬 만들기
            for j in range(PM_station_num) : # 매핑
                if(i==j) :  continue
#                i_lo = target_dataset.iloc[i,2]
#                i_la = target_dataset.iloc[i,1]
#                j_lo = target_dataset.iloc[j,2]
#                j_la = target_dataset.iloc[j,1]
                x1 = x 
                y1 = y
                x2 = int(PM_xy_hash_table[j][1])
                y2 = int(PM_xy_hash_table[j][2])
                
                e = 0.0001
                adj_distance[i][j] = 1/((x1-x2)**2+(y1-y2)**2)     #mydistance(i_lo,i_la,j_lo,j_la)
                adj_x[i][j] = 1/(x1-x2+e) if x1>x2 else 1/(x1-x2-e)
                adj_y[i][j] = 1/(y1-y2+e) if y1>y2 else 1/(y1-y2-e)

            sorted_weight = sorted(adj_distance[i], reverse=True) 
            for j in range(PM_station_num) : # 매핑
                if(i==j) :  continue     # 가장 큰 weight 8개만 남김
                if(sorted_weight[7]> adj[i][j]) :  
                    adj_x[i][j] = 0
                    adj_y[i][j] = 0
                    adj_distance[i][j] = 0
                    adj[i][j]  = 0
                else :
                    adj[i][j]  = 1
        for i in range(PM_station_num) :  # 무방향그래프화
             for j in range(PM_station_num) : # 상대좌표이므로 방향은 반대
                if(adj[i][j]!=0) : adj[j][i] = adj[i][j] 
                if(adj_distance[i][j]!=0) : adj_distance[j][i] = adj_distance[i][j]   
                if(adj_x[i][j]!=0) : adj_x[j][i] = -adj_x[i][j]  
                if(adj_y[i][j]!=0) : adj_y[j][i] = -adj_y[i][j]   
        adj_set.append(adj)
        adj_set.append(adj_distance)
        adj_set.append(adj_x)
        adj_set.append(adj_y)
    
 # evaluate a single model
def print_reference_value_npy(test, n_output_timestep, outputname, PM_station_num, PM_xy_hash_table, map_normal_max):
    if main.is_One_Station :
        test_np = np.zeros((len(test)-n_output_timestep+1, n_output_timestep , 1))
        for time_slice in range (len(test)-n_output_timestep+1):
            for timestep in range (n_output_timestep):
                        test_np[time_slice][timestep][0] =  map_normal_max[0]*test[time_slice+timestep, int(PM_xy_hash_table[main.One_Station][1]), 
                            int(PM_xy_hash_table[main.One_Station][2]), 0]  
    elif main.isMulti : 
        test_np = np.zeros((len(test)-n_output_timestep+1, n_output_timestep ,PM_station_num))
        for time_slice in range (len(test)-n_output_timestep+1):
            for timestep in range (n_output_timestep):
                for station_n in range(PM_station_num):
                        test_np[time_slice][timestep][station_n] =  map_normal_max[0]*test[time_slice+timestep, int(PM_xy_hash_table[station_n][1]), 
                            int(PM_xy_hash_table[station_n][2]), 0]      
    else :  # just one output
        test_np = np.zeros((len(test)-n_output_timestep+1, 1 ,PM_station_num)) 
        timestep = 0
        for time_slice in range (len(test)-n_output_timestep+1):
                for station_n in range(PM_station_num):   
                    test_np[time_slice][timestep][station_n] =  map_normal_max[0]*test[time_slice+n_output_timestep-1, int(PM_xy_hash_table[station_n][1]), 
                            int(PM_xy_hash_table[station_n][2]), 0]                
                              
    np.save(outputname, test_np)
    
os.environ["CUDA_VISIBLE_DEVICES"]='0'
    




PM_dataset_name = '미세먼지 측정소 주소_final_processed.csv'
Weather_dataset_name = '기상관측소 주소_final.csv'
original_data_name =  'combine_IDW.npy'
extractPollutantFeature = ['PM10','SO2','NO2']
pollutant_feature = len(extractPollutantFeature)
XY_feature = 2


def run() :
    # particulate matter
    print('--------0316-----------')
    print('train_set %d' % main.train_dataset_raw)
    print('val_set %d' % main.val_dataset_raw)
    print('test_set %d' % main.test_dataset_raw)
    print('input timestep %d' % main.input_timestep)
    print('output timestep %d' % main.output_timestep)
    print('epochs %d' % main.epochs)
    print('batch_size %d' % main.batch_size)
    print('-------------------')
    
    stations_data_map = np.load(original_data_name)
    
    if(main.val_dataset_raw+main.train_dataset_raw+main.test_dataset_raw>=stations_data_map.shape[0]):
        raise IndexError('Too much dataset.')
    
    stations_data_map0 = stations_data_map[:,:,:,9]
    PM_dataset, Weather_dataset, x_size, y_size, x_min, y_min = get_matrix(PM_dataset_name, Weather_dataset_name)
    
    PM_xy_hash_table, PM_station_num =  make_matrix(PM_dataset, x_min, y_min)
    np.save('stations_xy_hash_table', PM_xy_hash_table)
    
    Weather_xy_hash_table, Weather_station_num =  make_matrix(Weather_dataset, x_min, y_min)
    
    matrix_x = x_size
    matirx_y = y_size
    
    matrix_x = stations_data_map.shape[1]
    matirx_y = stations_data_map.shape[2]
    feature = stations_data_map.shape[3]
    
    # stations_data_map이 4차원이라 3차원으로 확인용
    
    """
    모델별 파라미터
    """
    # 먼저 reference 값 출력
    
    # ConvLSTM은 위치값을 가진 행렬을 포함한 5차원 데이터
    # [samples, time steps, rows, cols, channels]
    
    graph_adj_set = list()
    standard_graph_set = np.zeros((stations_data_map.shape[0], PM_station_num, feature)) 
    
    graphprocess(standard_graph_set, graph_adj_set, PM_dataset, stations_data_map,  PM_station_num, PM_xy_hash_table)


    # 기타 방식은 다른차원 데이터를 필요함
    
        # 3부터  temperature	  4 wind_s	5 humid	6 pressure	7 wind_sin	8 wind_cos	9 rain	10 day_temp_diff	  min_temp	season_0.0	season_1.0	season_2.0	season_3.0	hours_0.0	hours_1.0	hours_2.0	hours_3.0	hours_4.0	hours_5.0 x y
    
    #### Conv : 바람 feature만 씀
    conv_data = np.zeros((stations_data_map.shape[0],stations_data_map.shape[1],stations_data_map.shape[2],5))    
    conv_data[:,:,:,0] = stations_data_map[:,:,:,0] # PM10
    conv_data[:,:,:,1:3] = stations_data_map[:,:,:,7:9] # wind x y 
    conv_data[:,:,:,3] = stations_data_map[:,:,:,4]
    conv_data[:,:,:,4] = stations_data_map[:,:,:,5] # humid

    #### LSTM : 바람 sin, cos 제외라서 feature - 2
    lstm_data = np.zeros((stations_data_map.shape[0],stations_data_map.shape[1],stations_data_map.shape[2],feature-2))    
    lstm_data[:,:,:,:7] = stations_data_map[:,:,:,:7]
    lstm_data[:,:,:,7:] = stations_data_map[:,:,:,9:]
    
    #lstm_data[main.output_timestep:,:,:,-1] = stations_data_map[main.output_timestep:,:,:,0]-stations_data_map[:-main.output_timestep,:,:,0] # 차분값
    #lstm_data[0:main.output_timestep,:,:-1] = 0 # 차분값
    
    _,_, test = main.split_dataset(stations_data_map)
    
    
    # weather feature :온도, 풍속, 습도, 증기압, winds, windc, rain, day_temp, min_temp = 9ㄴ
    # date feature : 계절 4 + 시간 6 = 10
    dataset_2d = dataprocess_in_averages(9, 10, stations_data_map,  PM_station_num, Weather_station_num, PM_xy_hash_table, Weather_xy_hash_table)
    
    map_normal_max = normalized(stations_data_map)
    
    normalized(lstm_data)
    normalized(conv_data)
    
    normalized(dataset_2d)

    print_reference_value_npy(test, main.output_timestep, 'result_reference_value'+str(main.output_timestep)+'.csv', PM_station_num, PM_xy_hash_table, map_normal_max)
    
    
    
    print('main_ end %d' % stations_data_map.shape[0])
    
    repeat = str(main.output_timestep)
    
    dropout_rate = 0#0.5
    regularization = None#l2(1e-9)


#    instance_name = 'ver4-128' # China 4차원 ConvGRU
#    print(instance_name)
#    module_conv_develop_LSTM.evaluate_model(2,dropout_rate, regularization, regularization, 14, PM_xy_hash_table,128,False,False,3,original_data_name+instance_name+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
#
#
#    instance_name = 'ver5-128' # C를 중간에
#    print(instance_name)
#    module_conv_develop_LSTM.evaluate_model(2, dropout_rate, regularization, regularization, 20, PM_xy_hash_table,128,False,False,3,original_data_name+instance_name+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
#
 #    print('Sensor')  
#    module_conv_LSTM_sensor.evaluate_model(0, 0, None, None, 0, PM_xy_hash_table,256,False,False,3,original_data_name+'result_ver128SC'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
##

    print('ver.2')
    module_conv_develop_LSTM.evaluate_model(2,0, None, None, 5, PM_xy_hash_table,128,False,False,3,original_data_name+'result_verS256-2'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)


    instance_name = '' #cnn방식
    print(instance_name)
    module_conv_develop_LSTM.evaluate_model(2,dropout_rate, regularization, regularization, 6, PM_xy_hash_table,128,False,False,3,original_data_name+instance_name+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)





#
#    print('ver.1')
#    module_conv_develop_LSTM.evaluate_model(1, 0, None, None, 0, PM_xy_hash_table,128,False,False,3,original_data_name+'result_ver128-1'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
    
#    print('Sensor')  
#    module_conv_LSTM_sensor.evaluate_model(0, 0, None, None, 0, PM_xy_hash_table,128,False,False,3,original_data_name+'result_ver128SC'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
##
#    print('ver.simple')
#    module_conv_develop_simple_LSTM.evaluate_model(0, 0, None, None, 0, PM_xy_hash_table,128,False,False,3,original_data_name+'result_verS128-0'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
##    
#    print('ver.0')
#    module_conv_develop_LSTM.evaluate_model(0, 0, None, None, 0, PM_xy_hash_table,128,False,False,3,original_data_name+'result_ver128-0'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
#
#
###
#    print('ConvLSTM')  
#    module_conv_LSTM.evaluate_model(0, 0, None, None, 0, PM_xy_hash_table,128,False,False,3,original_data_name+'result_ver128C'+repeat, conv_data, lstm_data, map_normal_max, PM_station_num)
###
#

