#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: oneVSall.py
@time: 18-11-9 下午2:53
@desc:
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.utils.np_utils import to_categorical
import os
from wide_cnn_rnn import dichotomy
import config
import threading
from imblearn.under_sampling import RandomUnderSampler
seed = 8
np.random.seed(seed)

choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]
names=os.listdir(choose)
alphabet=[names[i][:-4] for i in range(len(names))]

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

cnn_x_name=["Seq_%d_y"%i for i in range(128)]
rnn_x_name=["Seq_%d_x"%i for i in range(128)]
filtered_train_data=pd.read_csv(config.HTTPS_CONFIG["all_train_path"])
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])

def train(c):
    #必须copy，否则会改变filtered_train
    train=filtered_train_data.copy()
    val=filtered_val_data.copy()
    train.loc[train["label"]==c,"label"]=-1
    train.loc[train["label"]!=-1,"label"]=1
    train.loc[train["label"]==-1,"label"]=0

    val.loc[val["label"]==c,"label"]=-1
    val.loc[val["label"]!=-1,"label"]=1
    val.loc[val["label"]==-1,"label"]=0
    # 训练集重采样数据
    # rus=RandomUnderSampler(random_state=0)
    # y=filtered_train_data["label"]
    # X=filtered_train_data[cnn_x_name+rnn_x_name]
    # X_train_data,y_train_data=rus.fit_sample(X,y)
    df1=train[train["label"]==0]
    print(str(c)+":"+str(df1.shape))
    df2=train[train["label"]==1].sample(frac=0.5)
    print(str(c)+":"+str(df2.shape))
    df=pd.concat([df1,df2],axis=0)
    print(str(c)+":"+str(df.shape))
    y_train_cnn=df["label"]
    X_train_cnn = df[cnn_x_name]
    y_train_cnn=to_categorical(y_train_cnn)
    X_train_cnn=np.asarray(X_train_cnn).reshape((-1,128,1))
    X_train_rnn = df[rnn_x_name]
    X_train_rnn=np.asarray(X_train_rnn)

    y_test_cnn=val["label"]
    X_test_cnn = val[cnn_x_name]
    y_test_cnn=to_categorical(y_test_cnn)
    X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))

    X_test_rnn = val[rnn_x_name]
    X_test_rnn=np.asarray(X_test_rnn)

    clf=dichotomy.cnn_rnn()
    X_tr = [X_train_cnn, X_train_rnn]
    Y_tr = y_train_cnn
    # 测试集
    X_te = [X_test_cnn, X_test_rnn]
    Y_te = y_test_cnn
    clf.fit(X_tr, Y_tr, epochs=10, batch_size=128)
    results = clf.evaluate(X_te, Y_te)
    print("model:%s--->"%str(c)+str(results))
    clf.save(config.HTTPS_CONFIG["models"]+str(c)+".h5")

if __name__ == '__main__':
    threads=[]
    # for c in range(config.HTTPS_CONFIG["num_class"]):
    #     t=threading.Thread(target=train,args=(c,))
    #     print ("start class:%s"%c)
    #     t.start()
    #     threads.append(t)
    # for k in threads:
    #     k.join()
    for c in range(config.HTTPS_CONFIG["num_class"]):
        train(c)