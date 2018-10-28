#!/usr/bin/env python
#coding=utf-8

import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Embedding, Reshape
from keras.layers import Flatten, merge, Dropout,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1_l2

# 全局变量
batch_size=128

# deep模型卷积参数
nb_filters=32
pool_size=(2,2)
kernel_size=(1,3)
cnn_input_shape=(1,64,1)
rnn_input_shape=(1,64,1)
rows,cols=1,64 # packet的数目，不足64的用0补齐为64

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

def wide(df_train, df_test, wide_cols, x_cols, target, model_type, method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test])

    train = df_wide[df_wide['IS_TRAIN'] == 1].drop(['IS_TRAIN'], axis=1)
    test = df_wide[df_wide['IS_TRAIN'] == 0].drop(['IS_TRAIN'], axis=1)

    # make sure all columns are in the same order and life is easier
    y_train = train.pop(target)
    y_train = np.array(y_train.values).reshape(-1, 1)
    X_train = train[wide_cols]
    y_test = test.pop(target)
    y_test = np.array(y_test.values).reshape(-1, 1)
    X_test = test[wide_cols]
    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    if model_type=='wide':
        pass
    else:
        return X_train, y_train, X_test, y_test

def cnn(df_train, df_test, cnn_cols, cont_cols, target, model_type, method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])

    train = df_deep[df_deep['IS_TRAIN'] == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep['IS_TRAIN'] == 0].drop('IS_TRAIN', axis=1)

    y_train = train.pop(target)
    y_train = np.array(y_train.values).reshape(-1, 1)
    X_train = np.array(train[cnn_cols]) / 256

    y_test = test.pop(target)
    y_test = np.array(y_test.values).reshape(-1, 1)
    X_test = np.array(test[cnn_cols]) / 256

    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    # 将负载数据进行reshape
    X_train_cnn = X_train.reshape(X_train.shape[0], rows, cols, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], rows, cols, 1)

    if model_type=='cnn':
        pass
    else:
        return X_train_cnn, y_train, X_test_cnn, y_test


def rnn(df_train, df_test, rnn_cols, cont_cols, target, model_type, method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])

    train = df_deep[df_deep['IS_TRAIN'] == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep['IS_TRAIN'] == 0].drop('IS_TRAIN', axis=1)

    y_train = train.pop(target)
    y_train = np.array(y_train.values).reshape(-1, 1)
    X_train = np.array(train[rnn_cols])

    y_test = test.pop(target)
    y_test = np.array(y_test.values).reshape(-1, 1)
    X_test = np.array(test[rnn_cols])

    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)
    from keras.preprocessing.sequence import pad_sequences
    # 将负载数据进行reshape
    X_train_rnn = pad_sequences(X_train,padding='post',maxlen=64)
    X_test_rnn = pad_sequences(X_test,padding='post',maxlen=64)

    if model_type=='rnn':
        pass
    else:
        return X_train_rnn, y_train, X_test_rnn, y_test


def rnn_cnn():
    X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn = []
    X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn = []

    # 训练集
    X_tr = [X_train_cnn, X_test_rnn]
    Y_tr = y_train_cnn
    # 测试集
    X_te = [X_test_cnn, X_test_rnn]
    Y_te = y_train_cnn

    # CNN： 使用CNN来表示序列特征，处理的是packet的长度序列
    cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
    # 两层卷积操作
    c = Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same', input_shape=cnn_input_shape)(cnn_inp)
    c = Activation('relu')(c)
    c = Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))(c)
    c = MaxPooling2D(pool_size=pool_size)(c)
    c = Dropout(0.25)(c)
    c = Flatten()(c)
    c = BatchNormalization()(c)
    # 1×100维
    c = Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

    # RNN: 输入是一个不定长的向量，单个元素的维度为8（通过embedding操作将message维度变为8）
    rnn_inp = Input(shape=rnn_input_shape, dtype='float32', name='rnn')
    # 输入是0-14编码，长度为15,输出的维度为8的连续值，每一个样本的长度为64,是packet最长的数目
    r = Embedding(15,8,64)(rnn_inp)
    r = SimpleRNN(64)(r)
    # cnn+rnn
    wcr_inp = concatenate([c, r])
    # wide特征和deep特征拼接，wide特征直接和输出节点相连
    wcr_out = Dense(Y_tr.shape[1], activation='softmax', name='wide_cnn_rnn')(wcr_inp)

    # 模型网络的入口和出口
    wcr = Model(inputs=[cnn_inp, rnn_inp], outputs=wcr_out)

    wcr.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics='accuracy')
    # 以下输入数据进行wide and deep模型的训练
    wcr.fit(X_tr, Y_tr, nb_epoch=100, batch_size=128)

    results = wcr.evaluate(X_te, Y_te)
    print("\n", results)

def wide_rnn_cnn():
    X_train_wide, y_train_wide, X_test_wide, y_test_wide=[]
    X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn=[]
    X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn=[]

    # 训练集
    X_tr = [X_train_wide,X_train_cnn,X_test_rnn]
    Y_tr = y_train_wide
    # 测试集
    X_te = [X_test_wide,X_test_cnn,X_test_rnn]
    Y_te = y_train_cnn

    # WIDE： 处理的是统计特征
    wide_inp = Input(shape=(X_train_wide.shape[1],), dtype='float32', name='wide')
    w=Dense()(wide_inp)

    # CNN： 使用CNN来表示序列特征，处理的是packet的长度序列
    cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
    # 两层卷积操作
    c = Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same', input_shape=cnn_input_shape)(cnn_inp)
    c = Activation('relu')(c)
    c = Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))(c)
    c = MaxPooling2D(pool_size=pool_size)(c)
    c = Dropout(0.25)(c)
    c = Flatten()(c)
    c = BatchNormalization()(c)
    # 1×100维
    c = Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

    # RNN: 输入是一个不定长的向量，单个元素的维度为8（通过embedding操作将message维度变为8）
    rnn_inp=Input(shape=rnn_input_shape,dtype='float32',name='rnn')
    r= SimpleRNN(32)(rnn_inp)


    # wide+cnn+rnn
    wcr_inp = concatenate([w, c, r])
    # wide特征和deep特征拼接，wide特征直接和输出节点相连
    wcr_out = Dense(Y_tr.shape[1], activation='softmax', name='wide_cnn_rnn')(wcr_inp)

    # 模型网络的入口和出口
    wcr = Model(inputs=[wide_inp, cnn_inp, rnn_inp], outputs=wcr_out)

    wcr.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics='accuracy')
    # 以下输入数据进行wide and deep模型的训练
    wcr.fit(X_tr, Y_tr, nb_epoch=100, batch_size=128)

    results = wcr.evaluate(X_te, Y_te)
    print("\n", results)