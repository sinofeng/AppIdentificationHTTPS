#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: overallEvaluate.py
@time: 18-11-9 下午4:54
@desci:
"""
import pandas as pd
import config
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import os
from result import figures

cnn_x_name=["Seq_%d_y"%i for i in range(128)]
rnn_x_name=["Seq_%d_x"%i for i in range(128)]
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])

y_test=filtered_val_data["label"]
# y_test=to_categorical(y_test)
X_test_cnn = filtered_val_data[cnn_x_name]
X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))
X_test_rnn = filtered_val_data[rnn_x_name]
X_test_rnn=np.asarray(X_test_rnn)

X_te = [X_test_cnn, X_test_rnn]
Y_te = y_test

outcomes=[]

models=os.listdir(config.HTTPS_CONFIG["models"])
for model in models:
    clf=load_model(config.HTTPS_CONFIG["models"]+model)
    outcome=clf.predict(X_te)
    outcomes.append(np.asarray(outcome)[:,0])
#转置
tmp=np.asarray(outcomes).T
predicts=[np.argmax(i) for i in tmp]
choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]
names=os.listdir(choose)
alphabet=[names[i][:-4] for i in range(len(names))]
#绘制混淆矩阵
figures.plot_confusion_matrix(Y_te, predicts,alphabet, "./")

