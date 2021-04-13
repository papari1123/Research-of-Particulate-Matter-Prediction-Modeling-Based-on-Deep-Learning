from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Layer
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, Conv3DTranspose, Conv2DTranspose
from keras.layers.convolutional import Conv3D
from keras.layers.local import LocallyConnected2D
from keras.layers.merge import concatenate, add, average, maximum
from keras.models import Model
from keras.layers.convolutional import MaxPooling1D
import json
import GRU_recurrent
from keras import regularizers
from keras import optimizers
from keras.regularizers import l2,l1, l1_l2
import keras
from keras.layers import LeakyReLU
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers.convolutional import AveragePooling2D
import keras.backend as K
from keras import losses

import main_val as main

class Dataprocess_layer(Layer):
    # 4차원 -> 3차원 
    def __init__(self, PM_xy_hash_table, n_out, **kwargs):
        self.PM_xy_hash_table = PM_xy_hash_table
        self.n_out = PM_xy_hash_table
        super(Dataprocess_layer, self).__init__(**kwargs)

    def call(self, X):
        if(main.is_One_Station is True): 
             input_shape_ = K.int_shape(X)
    #         out = K.placeholder((input_shape_[0], input_shape_[1], PM_station_num))
             out = list()
             x_   = int(self.PM_xy_hash_table[main.One_Station][1]) 
             y_   = int(self.PM_xy_hash_table[main.One_Station][2])
            #for i in range(maisn_output_timestep) : 
             out.append(X[:,:,x_,y_,:])
             out = K.reshape(out, (-1, input_shape_[1], 1, input_shape_[4]))
             return out

        else :
             PM_station_num = len(self.PM_xy_hash_table)
             input_shape_ = K.int_shape(X)
    #         out = K.placeholder((input_shape_[0], input_shape_[1], PM_station_num))
             out = list()
             for station_num in range(PM_station_num) : 
                 x_   = int(self.PM_xy_hash_table[station_num][1]) 
                 y_   = int(self.PM_xy_hash_table[station_num][2])
                #for i in range(maisn_output_timestep) : 
                 out.append(X[:,:,x_,y_,:])
             out = K.concatenate(out, axis=1)
             out = K.reshape(out, (-1, input_shape_[1], PM_station_num, input_shape_[4]))
             return out

    def compute_output_shape(self, input_shape):
        if(main.is_One_Station is True): 
            return (input_shape[0], input_shape[1], 1, input_shape[4])    
        else : 
            return (input_shape[0], input_shape[1], len(self.PM_xy_hash_table), input_shape[4])

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


# convert history into inputs and outputs
def to_supervised_china(train_china, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x,matrix_y ):
	# flatten data

	X_beijing= list()
	X_shanghai= list()
    
	in_start = 0
	# step over the entire history one time step at a time
	for train_id in range(len(train_china[0])):
		# define the end of the input sequence
		in_end = in_start + n_input_timestep
		in_start_china = in_end - n_output_timestep       
		out_end = in_end + n_output_timestep
		# ensure we have enough data for this instance
		if out_end < len(train_china[0]):

			xbeijing_input = train_china[0][in_start_china:in_end, :,:,:]
			xshanghai_input = train_china[1][in_start_china:in_end, :,:,:]
			# x_input = x_input.reshape((len(x_input), 1)) for univariable input
			X_beijing.append(xbeijing_input)
			X_shanghai.append(xshanghai_input)

            # move along one time step
		in_start += 1

    
	return array(X_beijing) , array(X_shanghai)



# convert history into inputs and outputs
def to_supervised(train_conv, train_lstm, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x,matrix_y ):
	# flatten data
	X_conv= list()
	X_lstm= list()

	if(main.isMulti is False): 
		n_out = 1 # 실제 아웃풋
		Y = np.zeros((len(train_conv)-n_input_timestep-n_output_timestep ,n_out,matrix_x, matrix_y))
	else : 
		n_out = n_output_timestep
		Y_rest = np.zeros((len(train_conv)-n_input_timestep-n_output_timestep ,n_out-1,matrix_x, matrix_y))
		Y_last = np.zeros((len(train_conv)-n_input_timestep-n_output_timestep ,1,matrix_x, matrix_y))

	in_start = 0
	# step over the entire history one time step at a time
	for train_id in range(len(train_conv)):
		# define the end of the input sequence
		in_end = in_start + n_input_timestep 
		out_end = in_end + n_output_timestep
		# ensure we have enough data for this instance
		if out_end < len(train_conv):
			x1_input = train_conv[in_start:in_end, :,:,:]
			x2_input = train_lstm[in_start:in_end, :,:,:]

			# x_input = x_input.reshape((len(x_input), 1)) for univariable input
			X_conv.append(x1_input)
			X_lstm.append(x2_input)

			if(main.is_One_Station is True): 
				for end_i in range(n_out):
                    ## 수정 필요
					Y[train_id,end_i,0] =  train_conv[in_end+end_i, int(PM_xy_hash_table[main.One_Station,1]), int(PM_xy_hash_table[main.One_Station,2]),0]
			elif(main.isMulti is False): 
				Y[train_id,0,:,:] =  train_conv[in_end+n_output_timestep-1,:,:,0] #- train_conv[in_end-1,:,:,0]
			else : 
				for end_i in range(n_out-1):
								Y_rest[train_id,end_i,:,:] =  train_conv[in_end+end_i,:,:,0] #- train_conv[in_end-1,:,:,0]
				Y_last[train_id,0,:,:] =   train_conv[in_end+n_output_timestep-1,:,:,0] #- train_conv[in_end-1,:,:,0]
                          
            # move along one time step
		in_start += 1
	if(main.isMulti is True) : Y = [Y_last ,Y_rest]
    
	return array(X_conv), array(X_lstm)  ,Y

# train the model
def build_model(PM_xy_hash_table, train_lstm, train_conv, train_china, val_lstm, val_conv, val_china, outputname, n_input_timestep, n_output_timestep, 
                kernel_size, matrix_x, matrix_y, PM_station_num, epochs, batch_size):
	# prepare data
	if(main.isMulti is False): 
		n_out = 1
	else : 
		n_out = n_output_timestep
	print('start supervising') 

	train_x1, train_x2,  train_y = to_supervised(train_conv, train_lstm, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x, matrix_y)
	val_x1, val_x2,  val_y = to_supervised(val_conv, val_lstm, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x, matrix_y)
	if(g_is_china != 5 and g_is_china != 6 and g_is_china != 7 and g_is_china > 0) :
		train_x3, train_x4 = to_supervised_china(train_china, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x, matrix_y)
		val_x3, val_x4= to_supervised_china(val_china, n_input_timestep, n_output_timestep, PM_station_num, PM_xy_hash_table, matrix_x, matrix_y)

	print('end supervising')
    
  
	# define parameters
	verbose = 1
	filter_n = g_filter
	# define model
	conv_input = Input(shape=(n_input_timestep, matrix_x, matrix_y, train_conv.shape[3]))
	lstm_input = Input(shape=(n_input_timestep, matrix_x, matrix_y, train_lstm.shape[3]))
	beijing_input = Input(shape=(n_output_timestep, matrix_x, matrix_y, train_china[0].shape[3]))
	shanghai_input = Input(shape=(n_output_timestep, matrix_x, matrix_y, train_china[1].shape[3]))
	if(g_is_china == 5 or g_is_china == 6 or g_is_china == 7) :
		train_x3 =  train_china[0][n_input_timestep-1:-n_output_timestep-1, :,:,:]
		val_x3 =   val_china[0][n_input_timestep-1:-n_output_timestep-1, :,:,:]
		train_x4 =  train_china[1][n_input_timestep-1:-n_output_timestep-1, :,:,:]
		val_x4 =   val_china[1][n_input_timestep-1:-n_output_timestep-1, :,:,:]
		beijing_input = Input(shape=(matrix_x, matrix_y, train_china[0].shape[3]))
		shanghai_input = Input(shape=(matrix_x, matrix_y, train_china[1].shape[3])) 

	conv_1 = conv_input
	lstm_1 = lstm_input
	if(g_version == 0) :
		lstm_encoder = ConvLSTM2D(filters=filter_n*2, padding= 'same', dropout= g_dropout , kernel_size=(5,5), return_sequences = True, activation='relu', kernel_regularizer = g_regular1)(lstm_1)
		mix_encoder = lstm_encoder
	else :
		conv_encoder = ConvLSTM2D(filters=filter_n, padding= 'same',dropout= g_dropout , kernel_size=(kernel_size,kernel_size), return_sequences = True, activation='relu', kernel_regularizer = g_regular1)(conv_1)
		lstm_encoder = ConvLSTM2D(filters=filter_n*2, padding= 'same', dropout= g_dropout , kernel_size=(5,5), return_sequences = True, activation='relu', kernel_regularizer = g_regular1)(lstm_1)
		mix_encoder = concatenate([conv_encoder, lstm_encoder])   
        
	mix_encoder_filter =  filter_n*2
	mix_encoder = ConvLSTM2D(filters=mix_encoder_filter, padding= 'same', dropout= g_dropout, kernel_size=(kernel_size,kernel_size), activation='relu', kernel_regularizer = g_regular2)(mix_encoder)
    #BATCH, X, Y, FILTER
#	if(g_skip is True) : 
#		concat1 = Lambda(lambda conv_input: conv_input[:,n_input_timestep-1,:,:,:], output_shape= ( matrix_x, matrix_y,train_conv.shape[3]))(conv_input)
#		concat2 = Lambda(lambda lstm_input: lstm_input[:,n_input_timestep-1,:,:,:], output_shape= ( matrix_x, matrix_y,train_lstm.shape[3]))(lstm_input)
#		mix_encoder = concatenate([mix_encoder, concat1, concat2])
	mix_decoder =mix_encoder
	mix_decoder = Flatten()(mix_encoder)
	mix_decoder = RepeatVector(n_output_timestep)(mix_decoder)
	mix_decoder = Reshape((n_output_timestep, matrix_x, matrix_y, mix_encoder_filter))(mix_decoder)
#    
	if(g_is_china == 10) :
#		b_input = Conv3DTranspose(filters = filter_n, kernel_size = (1,8,10), activation='relu', kernel_regularizer = g_regular1)(beijing_input)  
#		b_input = ConvLSTM2D(filters=filter_n, padding= 'same', dropout= g_dropout , kernel_size=(1,1),
#                              return_sequences = True, activation='relu', kernel_regularizer = g_regular2)(b_input)   
#		b_input = ConvLSTM2D(filters=filter_n, padding= 'same', dropout= g_dropout , kernel_size=(1,1),
#                              return_sequences = True, activation='relu', kernel_regularizer = g_regular2)(beijing_input)   
#		b_input = TimeDistributed(Conv2DTranspose(filters = filter_n, kernel_size = (8,10), activation='relu', kernel_regularizer = g_regular1))(b_input)  
		mix_decoder = concatenate([mix_decoder, b_input])              
	mix_decoder = ConvLSTM2D(filters=filter_n*2, padding= 'same', dropout= g_dropout , kernel_size=(kernel_size,kernel_size),
                              return_sequences = False, activation='relu', kernel_regularizer = g_regular2)(mix_decoder)   
        
        
#	mix_encoder = LeakyReLU(alpha=0.1)(mix_encoder)
    
    
#	lstm1 = Flatten()(lstm1)
#	lstm1 = RepeatVector(n_output_timestep)(lstm1)
#	lstm1 = Reshape((n_output_timestep, matrix_x, matrix_y, filter_n))(lstm1)
#	lstm1 = ConvLSTM2D(filters=filter_n, padding= 'same', kernel_size=(kernel_size,kernel_size), 
#                    kernel_regularizer=l2(5e-4), dropout = 0.2, recurrent_dropout = 0.3, activation='relu',  return_sequences = main.isMulti)(lstm1)
#
	adam_ = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00001, decay=0, amsgrad=True)
	if(main.is_One_Station is True): 
		mix_decoder = TimeDistributed(Conv2D(filters=filter_n*2, kernel_size= (5,5), activation='relu'))(mix_decoder)   
		mix_decoder = TimeDistributed(Conv2D(filters=filter_n*2, kernel_size= (5,5), activation='relu'))(mix_decoder)  
		mix_decoder = TimeDistributed(Conv2D(filters=1, kernel_size= (1,1), padding= 'same'))(mix_decoder) 
		output = Reshape((n_out, 1))(mix_decoder)
	elif(main.isMulti is False): # 스테이션 여러개, 아웃풋 한개    
		if(g_is_china == 5) :
				b_input = Conv2D(filters = filter_n, kernel_size = (1,1), activation='relu', kernel_regularizer = g_regular1)(beijing_input)
#				s_input = LocallyConnected2D(filters = filter_n, kernel_size = (1,1), activation='relu', kernel_regularizer = g_regular1)(shanghai_input)
				mix_decoder = concatenate([mix_decoder, b_input])
		elif(g_is_china == 6) :
				b_layer = Conv2D(filters = filter_n, kernel_size = (1,1), activation='relu', kernel_regularizer = g_regular1)(beijing_input)
#				s_layer = Conv2D(filters = filter_n, kernel_size = (1,1), activation='relu', kernel_regularizer = g_regular1)(shanghai_input) 
				mix_decoder = concatenate([mix_decoder, b_layer])   

            
            
#		mix_decoder = Conv2D(filters=filter_n*2, kernel_size= (1,1), activation='relu', kernel_regularizer = g_regular2)(mix_decoder)   
		mix_decoder = Conv2D(filters=filter_n*2, kernel_size= (1,1), activation='relu', kernel_regularizer = g_regular2)(mix_decoder)  
		mix_decoder = Conv2D(filters=1, kernel_size= (1,1))(mix_decoder) 
		output = Reshape((n_out, matrix_x, matrix_y))(mix_decoder)
		if(g_is_china>0) :
				model = Model(inputs=[conv_input, lstm_input, beijing_input, shanghai_input], outputs=output) 
		else :
				model = Model(inputs=[conv_input, lstm_input], outputs=output) 			
		model.compile(optimizer= adam_, loss= 'mse', metrics=['mae'])
	else :# 스테이션 여러개, 아웃풋 여러개
#		mix_decoder = TimeDistributed(Conv2D(filters=filter_n*2, kernel_size= (1,1), padding= 'same', activation='relu'))(mix_decoder)
#		mix_decoder = TimeDistributed(Conv2D(filters=filter_n*2, kernel_size= (1,1), padding= 'same', activation='relu'))(mix_decoder)
		mix_decoder = TimeDistributed(Conv2D(filters=1, kernel_size= (1,1), padding= 'same'))(mix_decoder)
		output_last = Lambda(lambda mix_decoder: mix_decoder[:,-1,:,:,0], output_shape= (1, matrix_x, matrix_y))(mix_decoder)
		output_rest = Lambda(lambda mix_decoder: mix_decoder[:,:-1,:,:,0], output_shape= (n_out-1, matrix_x, matrix_y))(mix_decoder)
		model = Model(inputs=[conv_input, lstm_input], outputs=[output_last, output_rest])     
		model.compile(optimizer= adam_, loss= ['mse','mse'], metrics=['mae','mae'],loss_weights=[1,1])
	model.summary()

	# fit network
	callbacks_list = [
            keras.callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 10,
                    ),
            keras.callbacks.ModelCheckpoint(
        filepath =  g_checkpoint_path,
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True,
        )
            ]

	if(main.isSaved is False) : 
				if(g_is_china>0) :
						hist = model.fit([train_x1, train_x2, train_x3, train_x4], train_y, epochs=epochs, batch_size=batch_size,  callbacks=   callbacks_list, validation_data=([val_x1, val_x2, val_x3, val_x4], val_y), verbose=verbose)
				else :
						hist = model.fit([train_x1, train_x2], train_y, epochs=epochs, batch_size=batch_size,  callbacks=   callbacks_list, validation_data=([val_x1, val_x2], val_y), verbose=verbose)
            
	model.load_weights(g_checkpoint_path)
	print('build finish with')
	return model

def update_model(model, x, y, val_x, val_y, batch_size):
	callbacks_list = [
            keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            ),
            keras.callbacks.ModelCheckpoint(
        filepath =  g_checkpoint_path,
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True
        )
            ]
	model.fit(x, y, epochs= 3,  callbacks=   callbacks_list, validation_data=(val_x, val_y), batch_size=batch_size, verbose=0)
	model.load_weights(g_checkpoint_path)

        
# make a forecast
def forecast(model, history_lstm, history_conv, history_beijing, history_shanghai  ,  map_normal_max, matrix_x, matrix_y):
	# flatten data
	data_lstm = array(history_lstm)
	data_conv = array(history_conv)
	data_beijing = array(history_beijing) 
	data_shanghai = array(history_shanghai)    
	#data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_lstm = data_lstm[-main.input_timestep:, :]
	input_conv = data_conv[-main.input_timestep:, :]

    
	if(g_is_china == 5 or g_is_china == 6 or g_is_china == 7) :
		input_beijing = data_beijing[-1, :]    
		input_beijing = input_beijing.reshape((1, matrix_x, matrix_y, input_beijing.shape[2]))
		input_shanghai = data_shanghai[-1, :]    
		input_shanghai = input_shanghai.reshape((1, matrix_x, matrix_y, input_shanghai.shape[2]))
	elif(g_is_china != 0) : 
		input_beijing = data_beijing[-main.output_timestep:, :]    
		input_beijing = input_beijing.reshape((1, main.output_timestep, matrix_x, matrix_y, input_beijing.shape[3]))
		input_shanghai = data_shanghai[-main.output_timestep:, :]    
		input_shanghai = input_shanghai.reshape((1, main.output_timestep, matrix_x, matrix_y, input_shanghai.shape[3]))
    
	# reshape into [samples, time steps, rows, cols, channels]
	input_conv = input_conv.reshape((1, main.input_timestep, matrix_x, matrix_y, input_conv.shape[3]))
	input_lstm = input_lstm.reshape((1, main.input_timestep, matrix_x, matrix_y, input_lstm.shape[3]))

	# forecast the next week
    
	if(g_is_china>0)      :
		yhat = model.predict([input_conv, input_lstm, input_beijing, input_shanghai],verbose=0)  
	else : 
		yhat = model.predict([input_conv, input_lstm],verbose=0)        
	# we only want the vector forecast
	return map_normal_max[0]* yhat[0]

def get_china_data (station_name, china_col) :
    
   
    baekryongdo_timestep = 6 # -6~ 0
    china_timestep = 24 # -25~ 0
    

    added_feature = china_timestep + baekryongdo_timestep*3
    
    
    china_dataset = np.load('china_processed.npy') 
    baekryongdo_dataset = np.load(station_name) 
    
    baekryongdo_size = baekryongdo_dataset.shape[3]
    
    combine_map_merged =  np.zeros((baekryongdo_dataset.shape[0], 
                                    baekryongdo_dataset.shape[1],
                                    baekryongdo_dataset.shape[2], 
                                    baekryongdo_size+added_feature))
    
    # 백령도 데이터 추가
    windx_col = 7 
    start = baekryongdo_size
    if(baekryongdo_size != 0) :
        combine_map_merged[:,:,:,:start] = baekryongdo_dataset
    # 아래는 백령도 과거 데이터 추가하고 싶으면 넣을 것.

#먼지 데이터    
    combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[0,0,0,0]
    
    for time in range (1,baekryongdo_timestep):
        combine_map_merged[time,:,:,start+baekryongdo_timestep-time:start+baekryongdo_timestep] = baekryongdo_dataset[0:time,0,0,0]
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep-time] = baekryongdo_dataset[0,0,0,0]   
    
    for time in range (baekryongdo_timestep, baekryongdo_dataset.shape[0]) :
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[time-baekryongdo_timestep:time,0,0,0]


    start = baekryongdo_size + baekryongdo_timestep
    combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[0,0,0,windx_col]
    
    for time in range (1,baekryongdo_timestep):
        combine_map_merged[time,:,:,start+baekryongdo_timestep-time:start+baekryongdo_timestep] = baekryongdo_dataset[0:time,0,0,windx_col]
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep-time] = baekryongdo_dataset[0,0,0,windx_col]   
    
    for time in range (baekryongdo_timestep, baekryongdo_dataset.shape[0]) :
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[time-baekryongdo_timestep:time,0,0,windx_col]

    start = baekryongdo_size + baekryongdo_timestep*2
    combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[0,0,0,windx_col+1]
    
    for time in range (1,baekryongdo_timestep):
        combine_map_merged[time,:,:,start+baekryongdo_timestep-time:start+baekryongdo_timestep] = baekryongdo_dataset[0:time,0,0,windx_col+1]
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep-time] = baekryongdo_dataset[0,0,0,windx_col+1]   
    
    for time in range (baekryongdo_timestep, baekryongdo_dataset.shape[0]) :
        combine_map_merged[time,:,:,start:start+baekryongdo_timestep] = baekryongdo_dataset[time-baekryongdo_timestep:time,0,0,windx_col+1]

    # 중국 데이터 추가
    start = baekryongdo_size+baekryongdo_timestep*3
    combine_map_merged[0,:,:,start:start+baekryongdo_timestep] = china_dataset[0,china_col]
    
    for time in range (1,china_timestep):
        combine_map_merged[time,:,:,start+china_timestep-time:start+china_timestep] = china_dataset[0:time,china_col]
        combine_map_merged[time,:,:,start:start+china_timestep-time] = china_dataset[0,china_col]
    
    
    for time in range (china_timestep, china_dataset.shape[0]) :
        combine_map_merged[time,:,:,start:start+china_timestep] = china_dataset[time-china_timestep:time,china_col]
    
    return  combine_map_merged#np.reshape(combine_map_merged[:,0,0,:], (combine_map_merged.shape[0],1,1,combine_map_merged.shape[3]))


g_regular1  = None
g_regular2  = None
g_checkpoint_path = 'nan'
g_filter = 1
g_terrain = False
g_skip = False
g_is_china = 0
g_dropout = 0
g_version = 0
# evaluate a single model
def evaluate_model(version, dropout, regular1, regular2, is_china, PM_xy_hash_table, n_filter, is_terrain, is_skip, kernel_size, outputname, conv_data, lstm_data,
                    map_normal_max, PM_station_num, model_ = None):
	global g_checkpoint_path, g_filter, g_terrain, g_skip, g_is_china, g_regular1, g_regular2, g_dropout, g_version
	g_regular1 = regular1
	g_regular2 = regular2
	g_filter = n_filter
	g_version = version
	g_is_china = is_china
	g_terrain = is_terrain
	g_skip = is_skip
	g_dropout = dropout
	g_checkpoint_path = 'cp'+ outputname + '.ckpt'
	# fit model
    
    #### China
	beijing_china_data = get_china_data('combine_IDW_Baekryongdo.npy',1)
	shanghai_china_data = get_china_data('combine_IDW_Jeju.npy',2)    
	normalized(beijing_china_data)
	normalized(shanghai_china_data)   
	china_data_list = []
	china_data_list.append(beijing_china_data)
	china_data_list.append(shanghai_china_data)
	train_lstm, val_lstm, test_lstm = main.split_dataset(lstm_data)
	train_conv, val_conv, test_conv = main.split_dataset(conv_data)
	train_china, val_china, test_china = main.split_dataset(china_data_list)
	matrix_x = test_conv.shape[1]
	matrix_y = test_conv.shape[2]
	if(model_ is None) :
		model = build_model(PM_xy_hash_table, train_lstm, train_conv, train_china, val_lstm, val_conv, val_china, outputname, main.input_timestep, main.output_timestep, kernel_size, matrix_x, matrix_y, PM_station_num, main.epochs, main.batch_size)
	else :
		model = model_  
#     history is a list of weekly data
	history_lstm = [x for x in val_lstm[-main.input_timestep:]]
	history_conv = [x for x in val_conv[-main.input_timestep:]]
	history_beijing = [x for x in val_china[0][-main.input_timestep:]]
	history_shanghai = [x for x in val_china[1][-main.input_timestep:]]
	"""
	walk-forward validation over each week
	"""
	predictions = list()
	print('start predict with Develop CONV LSTM')
	for i in range(len(test_conv)-main.output_timestep+1):
		# predict the week
		if((i%100) == 0) :
		  print('%d left' %(len(test_conv)-i))
		yhat_sequence = forecast(model, history_lstm, history_conv, history_beijing, history_shanghai, map_normal_max, matrix_x, matrix_y)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history_lstm.append(test_lstm[i, :])
		history_conv.append(test_conv[i, :])
		history_beijing.append(test_china[0][i, :])
		history_shanghai.append(test_china[1][i, :])   
		history_lstm.pop(0)
		history_conv.pop(0)
		history_beijing.pop(0)
		history_shanghai.pop(0)   
	# evaluate predictions days for each week
    # output file 저장
	np.save(outputname, predictions)