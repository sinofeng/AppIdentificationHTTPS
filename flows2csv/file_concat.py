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

choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]

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
# output=pd.DataFrame(columns=output_columns)
# output.to_csv(config.HTTPS_CONFIG["ouput_path"],index=False)
#
# record_type_names=["id"]+["Seq_"+str(i) for i in range(128)]+['label']
# record_type=pd.DataFrame(columns=record_type_names)
# record_type.to_csv(config.HTTPS_CONFIG["record_type_output_path"],index=False)
#
# packet_length_names=["id"]+["Seq_"+str(i) for i in range(128)]+['label']
# packet_length=pd.DataFrame(columns=packet_length_names)
# packet_length.to_csv(config.HTTPS_CONFIG["packet_length_output_path"],index=False)
#
# names=os.listdir(choose)
# softwares={names[i][:-4]:i for i in range(len(names))}


# for name in names:
#     # 合并统计特征
#     df1=pd.read_csv(choose+name)
#     df1['label']=softwares[name[:-4]]
#     df1.to_csv(config.HTTPS_CONFIG["ouput_path"],index=False,header=False,mode='a+')
#     del df1
#     # 合并 Record Type
#
#     df2=pd.read_csv(config.HTTPS_CONFIG["record_type_total"]+name[:-4]+"_record_type.csv").fillna(256)
#     df2['label']=softwares[name[:-4]]
#     df2.to_csv(config.HTTPS_CONFIG["record_type_output_path"],index=False,header=False,mode='a+')
#     del df2
#     # 合并 packet length
#
#     df3=pd.read_csv(config.HTTPS_CONFIG["packet_length_total"]+name[:-4]+"_packet_length.csv").fillna(0)
#     df3['label']=softwares[name[:-4]]
#     df3.to_csv(config.HTTPS_CONFIG["packet_length_output_path"],index=False,header=False,mode='a+')
#     del df3

output=pd.read_csv(config.HTTPS_CONFIG["ouput_path"])
train_index,val_index=train_test_split(range(output.__len__()),test_size=0.2,shuffle=True)

output_train,output_val=output.iloc[train_index],output.iloc[val_index]
output_train.to_csv(config.HTTPS_CONFIG["train_path"],index=False)
output_val.to_csv(config.HTTPS_CONFIG["val_path"],index=False)


record_type=pd.read_csv(config.HTTPS_CONFIG["record_type_output_path"])
train_index,val_index=train_test_split(range(record_type.__len__()),test_size=0.2,shuffle=True)

record_type_train,record_type_val=record_type.iloc[train_index],record_type.iloc[val_index]
record_type_train.to_csv(config.HTTPS_CONFIG["record_type_train_path"],index=False)
record_type_val.to_csv(config.HTTPS_CONFIG["record_type_val_path"],index=False)

packet_length=pd.read_csv(config.HTTPS_CONFIG["packet_length_output_path"])
packet_length_train,packet_length_val=packet_length.iloc[train_index],packet_length.iloc[val_index]
packet_length_train.to_csv(config.HTTPS_CONFIG["packet_length_train_path"],index=False)
packet_length_val.to_csv(config.HTTPS_CONFIG["packet_length_val_path"],index=False)
