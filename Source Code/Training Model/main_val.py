# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:25:32 2019

@author: LEE SEONGGU
"""

input_timestep = 0
output_timestep = 0
epochs = 0
batch_size = 0
is_One_Station = False
One_Station = 0
isSaved = False
isweightSave = False
isMulti = False




val_dataset_raw = 0
train_dataset_raw = 0
test_dataset_raw = 0

#하이퍼파라미터 튜닝시에는 test를 vaild로 사용#
def split_dataset(data):
    if(isinstance(data,list)) :
        vaild = list()
        train = list()
        test = list()
        for i in range(len(data)) :
            data_ele = data[i]
            train.append(data_ele[0:train_dataset_raw])
            vaild.append(data_ele[train_dataset_raw:val_dataset_raw + train_dataset_raw])
            test.append(data_ele[val_dataset_raw + train_dataset_raw:val_dataset_raw + train_dataset_raw + test_dataset_raw])
    else : 
        print('gg')
        train = data[0:train_dataset_raw] 
        vaild = data[train_dataset_raw:val_dataset_raw + train_dataset_raw]
        test =  data[val_dataset_raw + train_dataset_raw:val_dataset_raw + train_dataset_raw + test_dataset_raw]
    return train, vaild, test




