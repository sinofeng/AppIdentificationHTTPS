#/usr/bin/python envi
#coding=utf-8
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from result import figures
from joblib import dump, load
import input_w


train_data,train_labels,test_data,test_labels=input_w.inputs()


scoring = {'prec_macro': 'precision_macro','rec_micro': make_scorer(recall_score, average='macro')}
#留出法评估模型
clf=SVC(verbose=True,max_iter=2000)
clf.fit(train_data,train_labels)
dump(clf, './svm.model')
clf=load('./svm.model')

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
figures.plot_confusion_matrix(test_labels, predicts,alphabet, "./svm")
