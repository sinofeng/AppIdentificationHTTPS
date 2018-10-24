#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: file_concat.py
@time: 18-10-24 下午5:05
@desc: 修改choose选项进行分类别识别
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import config

choose=config.HTTPS_CONFIG["ebook_path"]

output_columns = ['id',
         'protocol_name',
         'src',
         'sport',
         'dst',
         'dport',
         'proto',
         'extension_servername_indication',
         'push_flag_ratio',
         'average_len',
         'average_payload_len',
         'pkt_count',
         'flow_average_inter_arrival_time',
         'kolmogorov',
         'shannon',
         'max_len',
         'min_len',
         'std_len',
         #'len_extension_signature_algorithms',
         'len_cipher_suites',
         'avrg_tcp_window',
         'max_tcp_window',
         'min_tcp_window',
         'var_tcp_window',
         'session_id_length',
         'avrg_ip_ttl',
         'max_ip_ttl',
         'min_ip_ttl',
         'label'
         ]
output=pd.DataFrame(columns=output_columns)
output.to_csv(config.HTTPS_CONFIG["ouput_path"],index=False)

names=os.listdir(choose)
softwares={names[i][:-4]:i for i in range(len(names))}

for name in names:
    df1=pd.read_csv(choose+name)
    df1['label']=softwares[name[:-4]]
    df1.to_csv(config.HTTPS_CONFIG["ouput_path"],index=False,header=False,mode='a+')
    del df1

output=pd.read_csv(config.HTTPS_CONFIG["ouput_path"])
output_train,output_val=train_test_split(range(output.__len__()),test_size=0.2,shuffle=True)
output_train,output_val=output.iloc[output_train],output.iloc[output_val]

output_train.to_csv(config.HTTPS_CONFIG["train_path"],index=False)
output_val.to_csv(config.HTTPS_CONFIG["val_path"],index=False)

del output_train,output_val