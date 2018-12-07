#/usr/bin/env python
#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn import preprocessing
from joblib import dump, load
from result import figures
import input_w
train_data,train_labels,test_data,test_labels=input_w.inputs()
train_data=preprocessing.MinMaxScaler().fit_transform(train_data)
test_data=preprocessing.MinMaxScaler().fit_transform(test_data)

lr=LogisticRegression(verbose=1,max_iter=500)
model=lr.fit(train_data,train_labels)


lr.fit(train_data,train_labels)
dump(lr, './lr.model')
clf=load('./lr.model')

predicts=clf.predict(test_data)
print(predicts)
print("accuracy_score:",accuracy_score(test_labels,predicts))
print("precision_score:",precision_score(test_labels,predicts,average='macro'))
# print("f1_score_micro:",f1_score(y_true,predicts,average='micro'))
print("f1_score_macro:",f1_score(test_labels,predicts,average='macro'))
# print("recall_score_micro:",recall_score(y_true,predicts,average='micro'))
print("recall_score_macro:",recall_score(test_labels,predicts,average='macro'))
# alphabet=["AIM","email","facebookchat","gmailchat","hangoutsaudio","hangoutschat","icqchat","netflix","skypechat","skypefile","spotify","vimeo","youtube","youtubeHTML5"]
alphabet=softwares=["Baidu Map",
                    "Baidu Post Bar",
                    "Netease cloud music",
                    "iQIYI",
                    "Jingdong",
                    "Jinritoutiao",
                    "Meituan",
                    "QQ",
                    "QQ music",
                    "QQ reader",
                    "Taobao",
                    "Weibo",
                    "CTRIP",
                    "Zhihu",
                    "Tik Tok",
                    "Ele.me",
                    "gtja",
                    "QQ mail",
                    "Tencent",
                    "Alipay"]
figures.plot_confusion_matrix(test_labels, predicts,alphabet, "./lr")