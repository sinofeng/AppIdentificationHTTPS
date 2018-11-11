#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: rnnOnly.py
@time: 18-11-10 下午5:30
@desc:
"""
from keras.layers import Input,concatenate
from keras.layers import Dense
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from result import figures
import os
from keras.utils.np_utils import to_categorical
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

y_train_rnn=filtered_train_data["label"]
X_train_rnn = filtered_train_data[rnn_x_name]
y_train_rnn=to_categorical(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn)

y_test_rnn=filtered_val_data["label"]
X_test_rnn = filtered_val_data[rnn_x_name]
y_test_rnn=to_categorical(y_test_rnn)
X_test_rnn=np.asarray(X_test_rnn)

batch_size=128


rnn_inp= Input(shape=(128,))
r=Embedding(257,16,input_length=128)(rnn_inp)
r=SimpleRNN(128,return_sequences=True)(r)
r=SimpleRNN(128)(r)
r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)

r = Dense(256,activation='relu')(r)
r = Dense(128,activation='relu')(r)
r_out = Dense(config.HTTPS_CONFIG["num_class"], activation='softmax', name='cnn_rnn')(r)

# 模型网络的入口和出口
cr = Model(inputs=[rnn_inp], outputs=r_out)
cr.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(cr.summary())

X_tr = [X_train_rnn]
Y_tr = y_train_rnn

# 测试集
X_te = [X_test_rnn]
Y_te = y_test_rnn

cr.fit(X_tr, Y_tr, epochs=100, batch_size=128)

results = cr.evaluate(X_te, Y_te)
predicts= cr.predict(X_te)
print(predicts)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")