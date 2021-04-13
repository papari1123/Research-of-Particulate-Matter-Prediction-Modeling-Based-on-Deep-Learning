# -*- coding: utf-8 -*-

"""
Created on Mon Sep 10 18:18:12 2018

: Keras 지원 라이브러리에서 ConvLSTM을 ConvGRU로 변경함.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.convolutional_recurrent import ConvRNN2D
import warnings
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.engine.base_layer import Layer


class ConvGRU2DCell(Layer):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
     
                 **kwargs):
        super(ConvGRU2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)

        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.reset_after = False
    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
#        if self.use_bias:
#            if self.unit_forget_bias:
#                def bias_initializer(_, *args, **kwargs):
#                    return K.concatenate([
#                        self.bias_initializer((self.filters,), *args, **kwargs),
#                        initializers.Ones()((self.filters,), *args, **kwargs),
#                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
#                    ])
#            else:
#                bias_initializer = self.bias_initializer
#            self.bias = self.add_weight(shape=(self.filters * 3,),
#                                        name='bias',
#                                        initializer=bias_initializer,
#                                        regularizer=self.bias_regularizer,
#                                        constraint=self.bias_constraint)
#        else:
#            self.bias = None
        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.filters,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.filters)
            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None 


#        self.kernel_i = self.kernel[:, :, :, :self.filters]
#        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
#        self.kernel_f = self.kernel[:, :, :, self.filters: self.filters * 2]
#        self.recurrent_kernel_f = (
#            self.recurrent_kernel[:, :, :, self.filters: self.filters * 2])
#        self.kernel_c = self.kernel[:, :, :, self.filters * 2: self.filters * 3]
#        self.recurrent_kernel_c = (
#            self.recurrent_kernel[:, :, :, self.filters * 2: self.filters * 3])
#        self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
#        self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

        self.kernel_z = self.kernel[:,:,:, :self.filters]
        self.recurrent_kernel_z = self.recurrent_kernel[:,:,:, :self.filters]
        # reset gate
        self.kernel_r = self.kernel[:, :,:,self.filters: self.filters * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,:,:,
                                                        self.filters:
                                                        self.filters * 2]
        # new gate
        self.kernel_h = self.kernel[:, :, :, self.filters * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, :, :, self.filters * 2:]

        if self.use_bias:
#            self.bias_i = self.bias[:self.filters]
#            self.bias_f = self.bias[self.filters: self.filters * 2]
#            self.bias_c = self.bias[self.filters * 2: self.filters * 3]
#            self.bias_o = self.bias[self.filters * 3:]
            self.input_bias_z = self.input_bias[:self.filters]
            self.input_bias_r = self.input_bias[self.filters: self.filters * 2]
            self.input_bias_h = self.input_bias[self.filters * 2:]
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.filters]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.filters: self.filters * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.filters * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None          
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state

        if 0 < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs 
            inputs_r = inputs 
            inputs_h = inputs 
        
                
        if 0 < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1


        x_z = self.input_conv(inputs_z, self.kernel_z, self.input_bias_z,
                              padding=self.padding)
        x_r = self.input_conv(inputs_r, self.kernel_r, self.input_bias_r,
                              padding=self.padding)
        x_h = self.input_conv(inputs_h, self.kernel_h, self.input_bias_h,
                              padding=self.padding)
        h_z = self.recurrent_conv(h_tm1_z,
                                  self.recurrent_kernel_z)
        h_r = self.recurrent_conv(h_tm1_r,
                                  self.recurrent_kernel_r)
        h_h = self.recurrent_conv(h_tm1_h,
                                  self.recurrent_kernel_h)
        if self.reset_after and self.use_bias:
            h_z = K.bias_add(h_z, self.recurrent_bias_z)
            h_r = K.bias_add(h_r, self.recurrent_bias_r)
            h_h = K.bias_add(h_h, self.recurrent_bias_h)
            
        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)
        h_r_h = r * h_h
        if self.reset_after and self.use_bias:
            h_r_h = K.bias_add(h_r_h, self.recurrent_bias_h)
            
        hh = self.activation(x_h + h_r_h)
        h = (1 - z) * hh  + z * h_tm1 
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
  
        return h, [h]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvGRU2D(ConvRNN2D):
  
    @interfaces.legacy_convlstm2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = ConvGRU2DCell(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              unit_forget_bias=unit_forget_bias,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout)
        super(ConvGRU2D, self).__init__(cell,
                                         return_sequences=return_sequences,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(ConvGRU2D, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
