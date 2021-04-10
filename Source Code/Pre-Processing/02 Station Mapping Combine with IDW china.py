
"""
Created on Mon Sep 10 18:18:12 2018

@author: Jaxx
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
"""
Station_Mapping 적용 후 사용
"""



"""
1. cell당 하나의 station만 남도록 추리고, 추린 station은 mapping함.
"""


  

                  
"""
3. 측정소 코드에 맞추어 데이터 변환
"""    
def weather_transFormat(weather_xy_hash_table, station_num) :
        standard_set = read_csv('./param/weather_standard_1418.csv', header=0, index_col = 0,  engine='python') 
        station_index = 0
        id_ = weather_xy_hash_table[station_index][0]
        file_num = 0
        print(station_index)
        for y in range(2014,2019):
            print(y)
            file_num = file_num +1
            if(file_num==1):
                dataset =read_csv('./data/SURFACE_ASOS_%d_HR_%d_%d_%d.csv' %(id_,y,y,y+1), header=0, index_col = 0, usecols =['일시','기온(°C)',
                                  '풍속(m/s)','풍향(16방위)','습도(%)','증기압(hPa)','강수량(mm)','적설(cm)'],  engine='python') 
            else:
                tempDataset = read_csv('./data/SURFACE_ASOS_%d_HR_%d_%d_%d.csv' %(id_,y,y,y+1), header=0, index_col = 0, usecols =['일시','기온(°C)',
                                  '풍속(m/s)','풍향(16방위)','습도(%)','증기압(hPa)','강수량(mm)','적설(cm)'],  engine='python') 
                dataset = concat([dataset,tempDataset])  
    #  5'기온(°C)', 6'풍속(m/s)', 7'습도(%)', 8'증기압(hPa)', 
        wind_s = np.zeros(dataset.shape[0])
        wind_v = np.zeros(dataset.shape[0])
        wind_cos = np.zeros(dataset.shape[0])
        wind_sin = np.zeros(dataset.shape[0])
              
        wind_s = dataset['풍속(m/s)'].values
        wind_v = dataset['풍향(16방위)'].values
             
        ttt = dataset.index.values
        date = pd.to_datetime(ttt)
        
        for i in range(dataset.shape[0]) :
            wind_sin[i] = wind_s[i] * math.sin(wind_v[i]*math.pi/180)
            wind_cos[i] = wind_s[i] * math.cos(wind_v[i]*math.pi/180)
        rain = dataset['강수량(mm)'].fillna(0).values
        snow = dataset['적설(cm)'].fillna(0).values
        water = rain + snow
        dataset.drop(columns=['풍향(16방위)'], inplace=True)
        dataset.drop(columns=['강수량(mm)'], inplace=True)
        dataset.drop(columns=['적설(cm)'], inplace=True)
        temp = dataset['기온(°C)'].values
        day_temp_diff = np.zeros(dataset.shape[0])
        min_temp = np.zeros(dataset.shape[0])   
        day_temp_diff[0] = 0
        min_temp[0] = temp[0]
        for i in range(1,dataset.shape[0]) :
                j = i-24 if i>=24 else 0
                day_temp_diff[i] = max(temp[j:i+1])-min(temp[j:i+1])
                min_temp[i] =  min(temp[j:i+1])
         #  9'wind_sin', 10'wind_cos', 
        dataset["wind_sin"] = wind_sin
        dataset["wind_cos"] = wind_cos
        dataset["rain"] = water
        dataset["day_temp_diff"] = day_temp_diff  
        dataset["min_temp"] = min_temp
        # 마지막에 포함시키거나 분리할 것.
#        dataset_weekday = pd.get_dummies(date.weekday, prefix  = 'weekday')  
        
        # season 처리
        month__  = date.month.values
        season__ = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]) :
          season__[i] =  0 if  month__[i]== 12 else  int((month__[i])/3)
          
        hour_  = date.hour.values
        hours__ = np.zeros(dataset.shape[0])    
        for i in range(dataset.shape[0]) :
          hours__[i] = int(hour_[i]/4)    
        
        dataset_month = pd.get_dummies(season__, prefix  = 'season')
        dataset_hour = pd.get_dummies(hours__, prefix  = 'hours')
        dataset_time = pd.concat([dataset_month, dataset_hour], axis=1)
        dataset_time.index = ttt
        dataset = pd.concat([dataset, dataset_time], axis=1)

        dataset.index.names = ['station_id']
        dataset = dataset.rename(columns={'일시':'date'})      
        dataset = dataset.rename(columns={'기온(°C)':'temperature'})         
        dataset = dataset.rename(columns={'풍속(m/s)':'wind_s'})
        dataset = dataset.rename(columns={'기온':'temperature'})
        dataset = dataset.rename(columns={'습도(%)':'humid'})   
        dataset = dataset.rename(columns={'증기압(hPa)':'pressure'})
        # 기상데이터는 row 개수가 측정소마다 달라 (예시로 18:30 때의 값이 들어감) 맞춰주는 작업
        
        result = pd.concat([standard_set,dataset],  join_axes=[standard_set.index], axis=1)
        result.to_csv('./output/weather%d.csv' %(id_) , header=True, index=True)

        return result.shape[1], result.shape[0]


def IDW_winddir(stations_map, stations_xy_hash_table, x_size, y_size, time) :
       # 보간 적용 for 미세먼지농도
    for x in range (x_size)    :
               for y in range (y_size)    :
                print('%d %d' % (x,y) )
                for data in range (int(len(stations_map)))    :
                    wind_intensity = stations_map[data,x,y,wind_col] 
                    if(stations_map[data,x,y,wind_sin_col] == nan_value or stations_map[data,x,y,wind_sin_col] == no_station_value) :
                          # 측정소가 있지만, 측정되지 않은 cell 또는 측정소가 없는 cell, sin_col은 대표
                          # 아래는 IDW적용, 먼저 벡터값 계산
                           wind_cos_element = 0
                           wind_sin_element = 0
                           for ff in [wind_sin_col, wind_cos_col] :
                              r_up = 0
                              r_down = 0.000000001
                              for x_ in range (x_size)    :
                                  for y_ in range (y_size)    :
                                      if(x_ == x and y_ == y) : continue
                                      distance_sq = ((x-x_)**2+(y-y_)**2)**2
                                      add_var = stations_map[data,x_,y_,ff]
                                      if(distance_sq>0 and add_var != nan_value and add_var != no_station_value) : 
                                          # 같은 cell이거나, 해당 더할 값이 이상하면 pass
                                             r_up += add_var/distance_sq # 분자
                                             r_down += 1/distance_sq # 분자  
                              if(ff==wind_cos_col): 
                                  wind_cos_element =  r_up/r_down
                              else :
                                  wind_sin_element =  r_up/r_down   
                              # 연산 끝, 에러 시 해당 timestep에서 모든 측정소 결과값이 없는 것임.  
                           r = (wind_sin_element**2 + wind_cos_element**2+0.00001)**0.5
                           ss = math.asin(wind_sin_element/r)
                           stations_map[data,x,y, wind_sin_col] = math.sin(ss)*wind_intensity
                           cc = math.acos(wind_cos_element/r)
                           stations_map[data,x,y, wind_cos_col] = math.cos(cc)*wind_intensity
    return stations_map
    

      
def IDW(stations_map, stations_xy_hash_table, x_size, y_size, time, start_feature, end_feature) :
       # 보간 적용 for 미세먼지농도
 
    for x in range (x_size)    :
               for y in range (y_size)    :
                  print('%d %d' % (x,y) )       
                  for ff in range (start_feature, end_feature)    :
                    #if(ff != wind_col) : continue
                    if(ff == wind_sin_col or ff == wind_cos_col) : continue  # 바람 방향 거름  """"""
                    for data in range (int(len(stations_map)))    :
                        if( stations_map[data,x,y,ff] == nan_value or stations_map[data,x,y,ff] == no_station_value) :
                              # 측정소가 있지만, 측정되지 않은 cell 또는 측정소가 없는 cell  
                              # 아래는 IDW적용
                              r_up = 0
                              r_down = 0.000000001
                              for x_ in range (x_size)    :
                                  for y_ in range (y_size)    :
                                      if(x_ == x and y_ == y) : continue
                                      distance_sq = ((x-x_)**2+(y-y_)**2)**2
                                      add_var = stations_map[data,x_,y_,ff]
                                      if(distance_sq>0 and add_var != nan_value and add_var != no_station_value) : 
                                          # 같은 cell이거나, 해당 더할 값이 이상하면 pass
                                             r_up += add_var/distance_sq # 분자
                                             r_down += 1/distance_sq # 분자  
                              # 연산 끝, 에러 시 해당 timestep에서 모든 측정소 결과값이 없는 것임. 
                              if(r_down==0) : stations_map[data,x,y,ff] = 0
                              else :          stations_map[data,x,y,ff] =r_up/r_down
                              
    return stations_map

    

def mapping_real_for_weather(map_, stations_xy_hash_table, x_size, y_size, feature, time) :

#    for station_index in range(len(stations_xy_hash_table)):
        ## station map에 맵핑
        station_index = 0
        print('mapping_real_for_weather')
        id_ = int(stations_xy_hash_table[station_index][0])
        dataset = read_csv('./output/weather%d.csv' %(id_) , header=0, index_col = 0, engine='python')
        dataset = dataset.fillna(method = 'ffill')
        # 값 넣기
        for i in range(time) : 
            for j in range(pollutant_feature,feature+pollutant_feature) :  # 오염물질 뒤에 차지
                map_[i][0][0][j] = dataset.iloc[i,j-pollutant_feature]  # 오염물질 뒤에 차지
        for i in range(time) :  
            print('%d ' % i)  
            for x_ in range(map_.shape[1]) : 
                for y_ in range(map_.shape[2]) : 

                        map_[i][x_][y_][pollutant_feature:feature+pollutant_feature] =  map_[i][0][0][pollutant_feature:feature+pollutant_feature]
        return    map_                     



def PM_mapping_with_average(map_, hashtable, x_size, y_size) : # 중복 x,y를 고려한 mapping
    # 측정소 미세먼지 값을 time, col, row, featrue 형식으로 변환
    PM_dataset = read_csv('./output/2014-2018 all_PM10_백령도.csv', header=0, index_col = 0, engine='python')
    PM_dataset = PM_dataset.fillna(method = 'ffill')
    for station_index in range(len(hashtable)):
            print(station_index)
            sub_data = PM_dataset[str(int(hashtable[station_index][0]))].values
            x = int(hashtable[station_index][1])
            y = int(hashtable[station_index][2])
            map_[:,x,y,0] = sub_data

    print('pm mapping end')
    
    return map_

def XY_feature_mapping(x_size,y_size,combine_map) :
    # 측정소 미세먼지 값을 time, col, row, featrue 형식으로 변환
    for x in range(x_size) :
            combine_map[:,x,:,pollutant_feature+weather_feature] = x-act_x
    for y in range(y_size) :             
            combine_map[:,:,y,pollutant_feature+weather_feature+1] = y
    return combine_map


"""
본 코드는 station_mapping 이라는 2015-2018 all_PM.csv 생성하는 코드 뒤에 실행되어야 함.
"""    
nan_value = -7777.1
no_station_value = -7778.1
extractPollutantFeature = ['PM10']
pollutant_feature = len(extractPollutantFeature)
XY_feature = 2
# 전제조건 : weather과 PM의  input_data (raw)가 동일해야 함. 

PM_dataset_name = './param/미세먼지 측정소 주소_99.csv'
Weather_dataset_name = './param/기상관측소 주소_99.csv'

PM_dataset_name_all = './param/미세먼지 측정소 주소_mini.csv'
Weather_dataset_name_all = './param/기상관측소 주소_mini.csv'


wind_sin_col = 5
wind_cos_col = 6
wind_col = 2

act_x = -3	
act_y = 9


x_size = 9
y_size = 9
x_min = 0
y_min = 0


def make_matrix(target_dataset, x_min, y_min) : 
    stations_map = np.zeros((x_size, y_size))  # station index가 행렬에 맵핑된다
    stations_xy_hash_table = np.zeros((target_dataset.shape[0],3))    
    using_stations_id = np.zeros(target_dataset.shape[0]) # 사용한 스테이션의 개수를 저장.
    use_station_num = 0
    for staion_index in range (target_dataset.shape[0]) :   # 그리드 매칭
                x_in = target_dataset.iloc[staion_index,3]-x_min
                y_in = target_dataset.iloc[staion_index,4]-y_min
                if True : # stations_map[x_in][y_in] == 0 :    # 중복 검사
                    stations_map[x_in][y_in] = target_dataset.iloc[staion_index,5]
                    using_stations_id[use_station_num]= target_dataset.iloc[staion_index,5]
                    use_station_num += 1

                    stations_xy_hash_table[staion_index][0] = int(stations_map[x_in][y_in])
                    stations_xy_hash_table[staion_index][1] = int(x_in)
                    stations_xy_hash_table[staion_index][2] = int(y_in)
    return stations_xy_hash_table, target_dataset.shape[0]

    # drop하기 (particulate stations used로 중복제거하여 변환)
PM_dataset =  read_csv(PM_dataset_name, header=0, index_col = 0, engine='python') 
Weather_dataset =     read_csv(Weather_dataset_name, header=0, index_col = 0, engine='python') 


weather_xy_hash_table, weather_station_num =  make_matrix(Weather_dataset, x_min, y_min )
PM_xy_hash_table, PM_station_num =  make_matrix(PM_dataset, x_min, y_min)
# 전제조건 : weather과 PM의  input_data (raw)가 동일해야 함.

weather_data_ex_skip = False

if weather_data_ex_skip is False :
    weather_feature, input_data =  weather_transFormat(weather_xy_hash_table, weather_station_num)
else :
    #  1 temperature	2 wind_s	3 humid	4 pressure	5 wind_sin	6 wind_cos	7 rain	day_temp_diff	min_temp	season_0.0	season_1.0	season_2.0	season_3.0	hours_0.0	hours_1.0	hours_2.0	hours_3.0	hours_4.0	hours_5.0
    weather_feature  =      19 
    input_data =        43823    
     
 # 미세먼지와 날씨를 포함한 실제 데이터가 맵핑된다.
combine_map = np.full((input_data, x_size, y_size, weather_feature + pollutant_feature + XY_feature),no_station_value)

# pm을 추가한다
combine_map = PM_mapping_with_average(combine_map, PM_xy_hash_table, x_size, y_size)

combine_map_TEST = combine_map[:,:,:,0]

# weather 정보 raw를 추가한다.
combine_map = mapping_real_for_weather(combine_map, weather_xy_hash_table, x_size, y_size , weather_feature, input_data)

print('pre-mapping end')
# weather interpolation 적용
#combine_map = IDW(combine_map, weather_xy_hash_table, x_size, y_size , input_data, pollutant_feature, pollutant_feature+ weather_feature)
#combine_map = IDW_winddir(combine_map, weather_xy_hash_table, x_size, y_size , input_data)
## pm interpolation 적용
#combine_map = IDW(combine_map, PM_xy_hash_table, x_size, y_size , input_data, 0, pollutant_feature)

# 위치좌표 X,Y추가한다.


combine_map = XY_feature_mapping(x_size,y_size,  combine_map)

#added_feature = 37 + 7 
#
#china_dataset = np.load('./output/china_processed.npy') 
#
#combine_map_merged =  np.zeros((input_data, x_size, y_size, weather_feature + pollutant_feature + XY_feature+added_feature))
#
## 차이나 자료 추가한다 -36~0까지





np.save('./output/combine_IDW_china_99', combine_map)
