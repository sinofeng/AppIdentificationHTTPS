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
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1_l2

# 全局变量
batch_size=128

# deep模型卷积参数
nb_filters=32
pool_size=(2,2)
kernel_size=(3,3)
input_shape=(64,64,1)
rows,cols=64,64

common_features=['push_flag_ratio','average_len','average_payload_len','pkt_count','flow_average_inter_arrival_time','kolmogorov','shannon']
payload_feature=["ss%s"%str(i) for i in range(4096)]
headers=common_features+payload_feature+['label']

def get_data(output_train,output_val,payload_train,payload_val):
    output_train=pd.read_csv(output_train)
    output_val=pd.read_csv(output_val)
    payload_train=pd.read_csv(payload_train)
    payload_val=pd.read_csv(payload_val)
    return output_train,output_val,payload_train,payload_val

# 定义特征交叉函数
def cross_columns(x_cols):
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


def val2idx(df, cols):
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.iteritems():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.iteritems():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, unique_vals


def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)

def continous_input(name):
    inp = Input(shape=(1,), dtype='float32', name=name)
    return inp, Reshape((1, 1))(inp)

# 定义模型wide,deep,wide and deep
def wide(df_train, df_test, wide_cols, x_cols, target, model_type, method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test])

    train = df_wide[df_wide['IS_TRAIN'] == 1].drop(['IS_TRAIN'], axis=1)
    test = df_wide[df_wide['IS_TRAIN'] == 0].drop(['IS_TRAIN'], axis=1)

    # make sure all columns are in the same order and life is easier
    y_train=train.pop(target)
    y_train = np.array(y_train.values).reshape(-1,1)
    X_train = train[wide_cols]
    y_test = test.pop(target)
    y_test = np.array(y_test.values).reshape(-1,1)
    X_test = test[wide_cols]

    # 多分类问题对标签进行one-hot编码
    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    # 使用sklearn对数据进行归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)

    if model_type == 'wide':

        activation, loss, metrics = fit_param[method]
        # metrics parameter needs to be passed as a list or dict
        if metrics:
            metrics = [metrics]

        # wide模型输入（在这里进行定义输入的形状）
        wide_inp = Input(shape=(X_train.shape[1],), dtype='float32', name='wide_inp')
        w = Dense(y_train.shape[1], activation=activation)(wide_inp)
        wide = Model(wide_inp, w)
        wide.compile(Adam(0.01), loss=loss, metrics=metrics)
        wide.fit(X_train, y_train, nb_epoch=100, batch_size=64)
        results = wide.evaluate(X_test, y_test)

        print ("\n", results)

    else:

        return X_train, y_train, X_test, y_test

def deep(df_train, df_test, deep_cols, cont_cols, target, model_type, method):

    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])

    train = df_deep[df_deep['IS_TRAIN'] == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep['IS_TRAIN'] == 0].drop('IS_TRAIN', axis=1)


    y_train = train.pop(target)
    y_train = np.array(y_train.values).reshape(-1,1)
    X_train = np.array(train[deep_cols])/256

    y_test = test.pop(target)
    y_test = np.array(y_test.values).reshape(-1,1)
    X_test = np.array(test[deep_cols])/256


    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    #将负载数据进行reshape
    X_train_deep=X_train.reshape(X_train.shape[0],rows,cols,1)
    X_test_deep=X_test.reshape(X_test.shape[0],rows,cols,1)

    if model_type == 'deep':

        activation, loss, metrics = fit_param[method]
        if metrics:
            metrics = [metrics]

        deep_inp = Input(shape=input_shape, dtype='float32', name='deep')
        d = Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]),padding='same',input_shape=input_shape)(deep_inp)
        d = Activation('relu')(d)
        d = Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]))(d)
        d = MaxPooling2D(pool_size=pool_size)(d)

        # d = Dropout(0.25)(d)
        d = Flatten()(d)
        d = BatchNormalization()(d)
        #d = Dense(100,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(d)
        d = Dense(256,activation='relu')(d)
        d = Dense(y_train.shape[1],activation='softmax')(d)
        # 将模型的头尾加入，完整的网络如同一个链式的结构一样被串接起来
        deep = Model(deep_inp, d)
        deep.compile(Adam(0.00001), loss=loss, metrics=metrics)
        # 在fit的时候将数据绑定进来
        deep.fit(X_train_deep, y_train, batch_size=64, nb_epoch=1000)

        #results = deep.evaluate(X_test_deep, y_test)
        results = deep.evaluate(X_train_deep, y_train)
        print ("\n", results)

    else:
        #返回的X_train和X_test需要进行reshape为32×32
        return X_train_deep, y_train, X_test_deep, y_test


def wide_deep(output_train, output_val,payload_train, payload_val,wide_cols, x_cols, deep_cols, embedding_cols, method):

    # Default model_type is "wide_deep"
    X_train_wide, y_train_wide, X_test_wide, y_test_wide = \
        wide(output_train, output_val,wide_cols, x_cols, target, model_type, method)

    X_train_deep, y_train_deep, X_test_deep, y_test_deep = \
        deep(payload_train, payload_val, deep_cols, embedding_cols,target, model_type, method)

    # 训练集
    X_tr_wd = [X_train_wide,X_train_deep]
    Y_tr_wd = y_train_deep  # wide or deep is the same here
    # 测试集
    X_te_wd = [X_test_wide,X_test_deep]
    Y_te_wd = y_test_deep  # wide or deep is the same here

    activation, loss, metrics = fit_param[method]
    if metrics:
        metrics = [metrics]

    # WIDE
    wide_inp = Input(shape=(X_train_wide.shape[1],), dtype='float32', name='wide')

    # DEEP: 输入的数据已经reshape过
    deep_inp = Input(shape=input_shape, dtype='float32', name='deep')


    # 两层卷积操作
    d = Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]),padding='same',input_shape=input_shape)(deep_inp)
    d = Activation('relu')(d)
    d = Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]))(d)
    d = MaxPooling2D(pool_size=pool_size)(d)
    d = Dropout(0.25)(d)
    d = Flatten()(d)
    d = BatchNormalization()(d)
    # 1×100维
    d = Dense(100,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(d)

    # WIDE + DEEP
    wd_inp = concatenate([wide_inp, d])
    # wide特征和deep特征拼接，wide特征直接和输出节点相连
    wd_out = Dense(Y_tr_wd.shape[1], activation=activation, name='wide_deep')(wd_inp)

    # 模型网络的入口和出口
    wide_deep = Model(inputs=[wide_inp,deep_inp], outputs=wd_out)

    wide_deep.compile(optimizer=Adam(lr=0.0001), loss=loss, metrics=metrics)
    # 以下输入数据进行wide and deep模型的训练
    wide_deep.fit(X_tr_wd, Y_tr_wd, nb_epoch=100, batch_size=128)

    results = wide_deep.evaluate(X_te_wd, Y_te_wd)
    print ("\n", results)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    # 用户使用参数
    ap.add_argument("--method", type=str, default="multiclass",help="fitting method")
    ap.add_argument("--model_type", type=str, default="deep",help="wide, deep or both")
    ap.add_argument("--output_train", type=str, default="../data/output_train.csv")
    ap.add_argument("--output_val", type=str, default="../data/output_val.csv")
    ap.add_argument("--payload_train", type=str, default="../data/payload_train.csv")
    ap.add_argument("--payload_val", type=str, default="../data/payload_val.csv")
    args = vars(ap.parse_args())
    method = args["method"]
    model_type = args['model_type']
    output_train = args['output_train']
    output_val = args['output_val']
    payload_train=args['payload_train']
    payload_val=args['payload_val']

    # 模型优化的参数
    fit_param = dict()
    fit_param['logistic']   = ('sigmoid', 'binary_crossentropy', 'accuracy')
    fit_param['regression'] = (None, 'mse', None)
    fit_param['multiclass'] = ('softmax', 'categorical_crossentropy', 'accuracy')

    output_train,output_val,payload_train,payload_val= get_data(output_train,output_val,payload_train,payload_val)

    # wide模型输入列
    wide_cols = common_features
    x_cols = ([], [])

    # deep模型输入列
    deep_cols = payload_feature
    embedding_cols = []

    # target for logistic
    target = 'label'
    # 训练模型并评估，输入为全部字段，使用*_cols来分wide和deep的列
    if model_type == 'wide':
        wide(output_train, output_val, wide_cols, x_cols, target, model_type, method)
    elif model_type == 'deep':
        deep(payload_train, payload_val, deep_cols, embedding_cols, target, model_type, method)
    else:
        wide_deep(output_train, output_val,payload_train, payload_val, wide_cols, x_cols, deep_cols,embedding_cols,  method)
