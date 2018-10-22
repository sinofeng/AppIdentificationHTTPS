#/usr/bin/env python
#coding=utf-8
import pandas as pd
import os
from sklearn.model_selection import train_test_split

output_folder_path='./output/'
payload3='./payload3/'

#start:定义数据的头部
#output_columns=['id','protocol_name','src','sport','dst','dport','proto','push_flag_ratio','average_len','average_payload_len','pkt_count','flow_average_inter_arrival_time','kolmogorov','shannon','max_len','min_len','std_len','label']

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
         'len_extension_signature_algorithms',
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
output.to_csv('output.csv',index=False)

del output
#end
#start:拼接文件,分别利用文件的名称给数据添加标签
names=os.listdir(output_folder_path)
softwares={"test1":0,"test2":1}
for name in names:
    df1=pd.read_csv(output_folder_path+name)
    df1['label']=softwares[name[:5]]
    df1.to_csv('output.csv',index=False,header=False,mode='a+')
    del df1
#end

#start:分割测试集和训练集合

output=pd.read_csv("./output.csv")
output_train,output_val=train_test_split(range(output.__len__()),test_size=0.2,shuffle=True)
output_train,output_val=output.iloc[output_train],output.iloc[output_val]
output_train.to_csv('./data/train.csv',index=False)
output_val.to_csv('./data/val.csv',index=False)
del output_train,output_val