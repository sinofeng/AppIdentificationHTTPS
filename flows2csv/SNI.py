# coding=utf-8
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
param = {
    'ngram_range' : (1, 1),
    'decode_error' : 'ignore',
    'token_pattern' : r'\b\w+\b',
    'analyzer' : 'char',
}
c = CountVectorizer(**param)
characters="abcdefghijklmnopqrstuvwxyz0123456789"
c.fit([characters])

output_train=pd.read_csv('../data/train.csv')
output_val=pd.read_csv('../data/val.csv')
train_labels=np.asarray(output_train.pop('label'),dtype=np.int32)
eval_labels=np.asarray(output_val.pop('label'),dtype=np.int32)

train_data=np.asarray(output_train["extension_servername_indication"])
train_data=c.transform(train_data)
train_data=train_data.toarray()
eval_data=np.asarray(output_val["extension_servername_indication"])
eval_data=c.transform(eval_data)
eval_data=eval_data.toarray()

# 如何将稀疏矩阵转换为正常的矩阵？？？
columns=["c_%d"%i for i in range(36)]

train_sni=pd.DataFrame(train_data,columns=columns)
train_sni.to_csv("./train_sni.csv",index=False)
del train_data
del train_sni

eval_sni=pd.DataFrame(eval_data,columns=columns)
eval_sni.to_csv("./eval_sni.csv",index=False)
del eval_data
del eval_sni
