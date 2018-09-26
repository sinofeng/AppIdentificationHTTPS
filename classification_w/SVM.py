#/usr/bin/python envi
#coding=utf-8
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,recall_score
from sklearn.model_selection import cross_validate

import input_w
train_data,train_labels,eval_data,eval_labels=input_w.inputs()


scoring = {'prec_macro': 'precision_macro','rec_micro': make_scorer(recall_score, average='macro')}
#留出法评估模型
clf1=SVC().fit(train_data,train_labels)
print(clf1.score(eval_data,eval_labels))

#交叉验证评估模型

clf2 = SVC()

scores2 = cross_val_score(clf2, train_data,train_labels, cv=5)
print("score:")
print(scores2)

scores3 = cross_validate(clf2, train_data,train_labels,scoring=scoring,cv=5, return_train_score=False)
print("recall score:")
print(scores3['test_rec_micro'])