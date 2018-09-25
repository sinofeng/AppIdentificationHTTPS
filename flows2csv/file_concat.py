#/usr/bin/env python
#coding=utf-8
import pandas as pd
import os
from sklearn.model_selection import train_test_split

output_folder_path='./output/'
payload_folder_path='./payload3/'

#start:定义数据的头部
output_columns=['id','protocol_name','src','sport','dst','dport','proto','push_flag_ratio','average_len','average_payload_len','pkt_count','flow_average_inter_arrival_time','kolmogorov','shannon','label']
output=pd.DataFrame(columns=output_columns)
output.to_csv('output.csv',index=False)
payload_columns_=["ss%s"%str(i) for i in range(4096)]
payload_columns=payload_columns_+['label']
payload=pd.DataFrame(columns=payload_columns)
payload.to_csv('payload.csv',index=False)
del output
del payload
#end
#start:拼接文件
names=os.listdir(output_folder_path)
softwares={"aiqiyi":0,"cloudmusic":1,"shoujibaidu":2}
for name in names:
    df1=pd.read_csv(output_folder_path+name)
    df1['label']=softwares[name[8:-20]]
    df1.to_csv('output.csv',index=False,header=False,mode='a+')
    del df1

    df2=pd.read_csv(payload_folder_path+name[:-9]+".csv",names=payload_columns_)
    df2['label']=softwares[name[8:-20]]
    df2.to_csv('payload.csv',index=False,header=False,mode='a+')
    del df2
#end

#start:分割测试集和训练集合

output=pd.read_csv("./output.csv")
output_train,output_val=train_test_split(range(output.__len__()),test_size=0.2,shuffle=True)
print(output_train)
print(output_val)
output.iloc[output_train].to_csv('../data/output_train.csv',index=False)
output.iloc[output_val].to_csv('../data/output_val.csv',index=False)
del output

payload=pd.read_csv("./payload.csv")
payload_train,payload_val=payload.iloc[output_train],payload.iloc[output_val]
payload_train.to_csv('../data/payload_train.csv',index=False)
payload_val.to_csv('../data/payload_val.csv',index=False)
del payload
del payload_train,payload_val