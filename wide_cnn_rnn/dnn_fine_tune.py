#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: dnn_fine_tune.py
@time: 18-11-28 下午5:23
@desc:
"""
from keras.layers import Input
from keras.layers import Dense
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder
from result import figures
import os

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

dnn_x_name=["x_%s"%str(i) for i in range(128)]
train=pd.read_csv("../result/train.csv")
test=pd.read_csv("../result/test.csv")

y_train_dnn=train["label"]
X_train_dnn = train[dnn_x_name]
y_train_dnn=np.asarray(y_train_dnn).reshape(-1,1)
y_train_dnn=onehot(y_train_dnn)
X_train_dnn=np.asarray(X_train_dnn)

y_test_dnn=test["label"]
X_test_dnn = test[dnn_x_name]
y_test_dnn=np.asarray(y_test_dnn).reshape(-1,1)
y_test_dnn=onehot(y_test_dnn)
X_test_dnn=np.asarray(X_test_dnn)


batch_size=128

dnn_inp=Input(shape=(128,))

d_out = Dense(14, activation='softmax', name='dnn_cnn_rnn')(dnn_inp)

# 模型网络的入口和出口
d = Model(inputs=dnn_inp, outputs=d_out)
d.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(d.summary())

X_tr = X_train_dnn
Y_tr = y_train_dnn
# 测试集
X_te = X_test_dnn
Y_te = y_test_dnn
d.fit(X_tr, Y_tr, epochs=10000, batch_size=32)

results = d.evaluate(X_te, Y_te)
print("\n", results)
results = d.evaluate(X_te, Y_te)
predicts= d.predict(X_te)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)

alphabet=["AIM","email","facebookchat","gmailchat","hangoutsaudio","hangoutschat","icqchat","netflix","skypechat","skypefile","spotify","vimeo","youtube","youtubeHTML5"]
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")