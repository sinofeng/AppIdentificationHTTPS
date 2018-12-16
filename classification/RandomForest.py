#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: RandomForest.py
@time: 18-12-7 下午7:16
@desc:
"""
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from result import figures
from joblib import dump, load
import input_w
from sklearn.ensemble import RandomForestClassifier
train_data,train_labels,test_data,test_labels=input_w.inputs()
clf = RandomForestClassifier(n_estimators=100, max_depth=12,random_state=0,verbose=1)
clf.fit(train_data,train_labels)

dump(clf, './randomForest.model')
clf=load('./randomForest.model')

predicts=clf.predict(test_data)
print(predicts)
print("predicts:",len(predicts))
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
figures.plot_confusion_matrix(test_labels, predicts,alphabet, "./rf")