#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: dnnOnly.py
@time: 18-11-10 下午6:20
@desc:
"""
from keras.layers import Input,concatenate
from keras.layers import Dense
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from result import figures
import os
choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]
names=os.listdir(choose)
alphabet=[names[i][:-4] for i in range(len(names))]

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

dnn_x_name=['push_flag_ratio',
            'average_len',
            'average_payload_len',
            'pkt_count',
            'flow_average_inter_arrival_time',
            'kolmogorov',
            'shannon',
            'max_len',
            'min_len',
            'std_len',
            'len_cipher_suites',
            'avrg_tcp_window',
            'max_tcp_window',
            'min_tcp_window',
            'var_tcp_window',
            #'session_id_length',
            'avrg_ip_ttl',
            'max_ip_ttl',
            'min_ip_ttl']
cnn_x_name=["Seq_%d_y"%i for i in range(128)]
rnn_x_name=["Seq_%d_x"%i for i in range(128)]
filtered_train_data=pd.read_csv(config.HTTPS_CONFIG["all_train_path"])
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])

y_train_dnn=filtered_train_data["label"]
X_train_dnn = filtered_train_data[dnn_x_name]
y_train_dnn=np.asarray(y_train_dnn).reshape(-1,1)
y_train_dnn=onehot(y_train_dnn)
X_train_dnn=np.asarray(X_train_dnn)
y_test_dnn=filtered_val_data["label"]
X_test_dnn = filtered_val_data[dnn_x_name]
y_test_dnn=np.asarray(y_test_dnn).reshape(-1,1)
y_test_dnn=onehot(y_test_dnn)
X_test_dnn=np.asarray(X_test_dnn)


batch_size=128

nb_filters=32
kernel_size=8

dnn_inp=Input(shape=(18,))
d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)
d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)
d = Dense(32,activation='relu')(d)
d_out = Dense(config.HTTPS_CONFIG["num_class"], activation='softmax', name='dnn_cnn_rnn')(d)

# 模型网络的入口和出口
d = Model(inputs=[dnn_inp], outputs=d_out)
d.compile(optimizer=Adam(lr=0.01),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(d.summary())

X_tr = [X_train_dnn]
Y_tr = y_train_dnn
# 测试集
X_te = [X_test_dnn]
Y_te = y_test_dnn
d.fit(X_tr, Y_tr, epochs=100, batch_size=128)

results = d.evaluate(X_te, Y_te)
print("\n", results)
predicts= d.predict(X_te)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")