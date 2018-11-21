#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: dichotomy.py
@time: 18-11-9 下午2:39
@desc:二分类模型，one-vs-all策略
"""
from keras.layers import Input,concatenate
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Multiply
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.optimizers import Adam
import config


batch_size=128

nb_filters=32
kernel_size=8

def cnn_rnn():
    cnn_input_shape=(128,1)
    cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
    # 两层卷积操作
    c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
    c = MaxPooling1D()(c)
    c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(c)
    c = MaxPooling1D()(c)
    c = Flatten()(c)
    c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
    c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
    rnn_inp= Input(shape=(128,))
    r=Embedding(257,16,input_length=128)(rnn_inp)
    r=SimpleRNN(128,return_sequences=True)(r)
    r=SimpleRNN(128)(r)
    # r=LSTM(128,return_sequences=True)(r)
    # r=LSTM(128)(r)
    r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)
    # r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)
    cr_inp = concatenate([c, r])
    # 加入注意力机制
    attention_probs=Dense(64,activation='softmax',name="attention_probs")(cr_inp)
    cr_inp=Multiply()([cr_inp,attention_probs])
    # wide特征和deep特征拼接，wide特征直接和输出节点相连
    cr = Dense(256,activation='relu')(cr_inp)
    cr = Dense(128,activation='relu')(cr)
    cr_out = Dense(2, activation='softmax', name='cnn_rnn')(cr)
    # 模型网络的入口和出口
    cr = Model(inputs=[cnn_inp, rnn_inp], outputs=cr_out)
    cr.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
    # 以下输入数据进行wide and deep模型的训练
    print(cr.summary())
    return cr

def dnn_rnn_cnn():
    dnn_inp=Input(shape=(18,))
    d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)
    d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)
    cnn_input_shape=(128,1)
    cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
    # 两层卷积操作
    c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
    c = MaxPooling1D()(c)
    c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(c)
    c = Flatten()(c)
    c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
    rnn_inp= Input(shape=(128,))
    r=Embedding(257,16,input_length=128)(rnn_inp)
    r=SimpleRNN(128,return_sequences=True)(r)
    r=SimpleRNN(128)(r)
    r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)
    dcr_inp = concatenate([d, c, r])
    # 加入注意力机制
    attention_probs=Dense(96,activation='softmax',name="attention_probs")(dcr_inp)
    dcr_inp=Multiply()([dcr_inp,attention_probs])
    # wide特征和deep特征拼接，wide特征直接和输出节点相连
    dcr = Dense(32,activation='relu')(dcr_inp)
    dcr_out = Dense(2, activation='softmax', name='dnn_cnn_rnn')(dcr)
    # 模型网络的入口和出口
    dcr = Model(inputs=[dnn_inp,cnn_inp, rnn_inp], outputs=dcr_out)
    dcr.compile(optimizer=Adam(lr=0.01),loss="categorical_crossentropy",metrics=["accuracy"])
    # 以下输入数据进行wide and deep模型的训练
    print(dcr.summary())
    return dcr