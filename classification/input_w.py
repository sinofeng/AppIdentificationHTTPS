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
    output_train=pd.read_csv(config.HTTPS_CONFIG["train_data_sni_path"])
    output_val=pd.read_csv(config.HTTPS_CONFIG["val_data_sni_path"])
    # output_train=pd.read_csv('../data/payload_train.csv')
    # output_val=pd.read_csv('../data/payload_val.csv')
    common_features=['push_flag_ratio',
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
                     'min_ip_ttl',
                     ] #只使用commonfeatures
    columns=["c_%d"%i for i in range(36)]
    # common_features=columns # 只使用SNI
    common_features=common_features+columns #混合使用
    train_labels=np.asarray(output_train.pop('label'),dtype=np.int32)
    train_data=np.asarray(output_train[common_features],dtype=np.float32)
    # train_data=np.asarray(output_train,dtype=np.float32)
    eval_labels=np.asarray(output_val.pop('label'),dtype=np.int32)
    eval_data=np.asarray(output_val[common_features],dtype=np.float32)
    # eval_data=np.asarray(output_val,dtype=np.float32)
    return train_data,train_labels,eval_data,eval_labels