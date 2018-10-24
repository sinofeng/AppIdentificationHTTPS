#/usr/bin/python envi
#coding=utf-8
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np
import os
import input_w
import config
from result import figures
# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(17, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True

train_data,train_labels,eval_data,eval_labels=input_w.inputs()

# lgb 参数
params={
    "learning_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": 5,
    "objective": "multiclass",
    "num_class": config.HTTPS_CONFIG["num_class"],
    "silent": True,
    "verbosity": -1
}

train_data = lgb.Dataset(train_data, label=train_labels)
validation_data = lgb.Dataset(eval_data, label=eval_labels)

clf=lgb.train(params,train_data,num_boost_round=100000, valid_sets=[validation_data],early_stopping_rounds=500,feval=f1_score_vali,verbose_eval=1)

names=os.listdir(config.HTTPS_CONFIG["processed_data"])
labels=[names[i][:-4] for i in range(len(names))]

y_pred=clf.predict(eval_data)
y_pred=[np.argmax(i) for i in y_pred]
eval_labels=[i for i in eval_labels]

figures.plot_confusion_matrix(eval_labels,y_pred,labels,config.HTTPS_CONFIG["outcome"]+'lgb_confusion_matrix.png')