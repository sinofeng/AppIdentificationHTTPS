#/usr/bin/env python
#coding=utf-8
import pandas as pd
import numpy as np
import config
def inputs():
    """
    commonfeatures控制训练的特征输入
    :return:
    """
    output_train=pd.read_csv("../../data/train.csv")
    output_val=pd.read_csv("../../data/test.csv")
    # output_train=pd.read_csv('../data/payload_train.csv')
    # output_val=pd.read_csv('../data/payload_val.csv')
    common_features=["s_%d"%i for i in range(24)]

    train_labels=np.asarray(output_train.pop('label'),dtype=np.int32)
    train_data=np.asarray(output_train[common_features],dtype=np.float32)
    # train_data=np.asarray(output_train,dtype=np.float32)
    eval_labels=np.asarray(output_val.pop('label'),dtype=np.int32)
    eval_data=np.asarray(output_val[common_features],dtype=np.float32)
    # eval_data=np.asarray(output_val,dtype=np.float32)
    return train_data,train_labels,eval_data,eval_labels