# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:31:28 2019

"""

import main_val as main

import main_m


"""
전 모델 공통 파라미터
"""
#input timestep

main.epochs = 100
main.batch_size = 128
main.isMulti = False
main.is_One_Station = False
main.One_Station = 31
main.isSaved = False
#main.isweightSave = True




# batch는 val과 train모두에게 나누어 떨어져야 함
main.val_dataset_raw = int(365*24)# + main.input_timestep  +main.output_timestep
main.train_dataset_raw = int(365*24*3)# + main.input_timestep  +main.output_timestep
main.test_dataset_raw = int(365*24)# + main.input_timestep  +main.output_timestep
#
input_timestep = [1,4,12,24]
main.input_timestep = 24
for main.output_timestep in input_timestep:
    main_m.run()