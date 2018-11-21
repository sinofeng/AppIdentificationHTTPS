#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: cnnOnly.py
@time: 18-11-10 下午5:30
@desc:
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

cnn_x_name=["p_%d"%i for i in range(1024)]
filtered_train_data=pd.read_csv(config.HTTPS_CONFIG["packet_payload_train_path"])
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["packet_payload_val_path"])

y_train_cnn=filtered_train_data["label"]
X_train_cnn = filtered_train_data[cnn_x_name]
# y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
# y_train_cnn=onehot(y_train_cnn)
y_train_cnn=to_categorical(y_train_cnn)
X_train_cnn=np.asarray(X_train_cnn).reshape((-1,1024,1))

# print(X_train_rnn)

y_test_cnn=filtered_val_data["label"]
X_test_cnn = filtered_val_data[cnn_x_name]
# y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
# y_test_cnn=onehot(y_test_cnn)
y_test_cnn=to_categorical(y_test_cnn)
X_test_cnn=np.asarray(X_test_cnn).reshape((-1,1024,1))



batch_size=128

nb_filters=128
kernel_size=3
cnn_input_shape=(1024,1)
cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
# 两层卷积操作
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
c = MaxPooling1D()(c)
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(c)
c = MaxPooling1D()(c)
c = Flatten()(c)

# c = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
c = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
c = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

c_out = Dense(config.HTTPS_CONFIG["num_class"], activation='softmax', name='cnn_rnn')(c)

# 模型网络的入口和出口
c = Model(inputs=[cnn_inp,], outputs=c_out)
c.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(c.summary())

X_tr = [X_train_cnn]
Y_tr = y_train_cnn

# 测试集
X_te = [X_test_cnn]
Y_te = y_test_cnn

c.fit(X_tr, Y_tr, epochs=100, batch_size=128)

results = c.evaluate(X_te, Y_te)
predicts= c.predict(X_te)
print(predicts)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")