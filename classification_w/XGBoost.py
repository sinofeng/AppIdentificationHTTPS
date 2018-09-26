#/usr/bin/python envi
#coding=utf-8
import xgboost as xgb
from xgboost import XGBClassifier
import input_w
from result import figures
params={
    'max_depth':12,
    'learning_rate':0.05,
    'n_estimators':752,
    'silent':True,
    'objective':"multi:softmax",
    'nthread':4,
    'gamma':0,
    'max_delta_step':0,
    'subsample':1,
    'colsample_bytree':0.9,
    'colsample_bylevel':0.9,
    'reg_alpha':1,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'base_score':0.5,
    'seed':2018,
    'missing':None,
    'num_class':3,
    'verbose':1,
    'eval_metric':'mlogloss'
}

plst = list(params.items())
num_rounds = 5000 # 迭代次数
train_data,train_labels,eval_data,eval_labels=input_w.inputs()

xgb_train=xgb.DMatrix(train_data,label=train_labels)
xgb_val=xgb.DMatrix(eval_data,eval_labels)
watchlist = [(xgb_val, 'val')]
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
