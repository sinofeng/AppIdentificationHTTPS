#/usr/bin/env python
#coding=utf-8
import pandas as pd
import numpy as np

def inputs():
    output_train=pd.read_csv('../data/output_train.csv')
    output_val=pd.read_csv('../data/output_val.csv')
    # output_train=pd.read_csv('../data/payload_train.csv')
    # output_val=pd.read_csv('../data/payload_val.csv')
    common_features=['push_flag_ratio','average_len','average_payload_len','pkt_count','flow_average_inter_arrival_time','kolmogorov','shannon']

    train_labels=np.asarray(output_train.pop('label'),dtype=np.int32)
    train_data=np.asarray(output_train[common_features],dtype=np.float32)
    # train_data=np.asarray(output_train,dtype=np.float32)
    eval_labels=np.asarray(output_val.pop('label'),dtype=np.int32)
    eval_data=np.asarray(output_val[common_features],dtype=np.float32)
    # eval_data=np.asarray(output_val,dtype=np.float32)
    return train_data,train_labels,eval_data,eval_labels