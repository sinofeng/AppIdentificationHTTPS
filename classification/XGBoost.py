#/usr/bin/python envi
#coding=utf-8
import xgboost as xgb
from xgboost import XGBClassifier
import input_w
from result import figures
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
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
    'num_class':20,
    'verbose':1,
    'eval_metric':'mlogloss'
}

plst = list(params.items())
num_rounds = 50 # 迭代次数
train_data,train_labels,test_data,test_labels=input_w.inputs()

xgb_train=xgb.DMatrix(train_data[500:],label=train_labels[500:])
xgb_val=xgb.DMatrix(train_data[:500],train_labels[:500])
xgb_test=xgb.DMatrix(test_data)


watchlist = [(xgb_val, 'val')]
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model("./xgb.model")

bst=xgb.Booster({'nthread':4})
bst.load_model('./xgb.model')
preds = bst.predict(xgb_test)
print(preds.tolist())
print("accuracy_score:",accuracy_score(test_labels,preds))
print("precision_score:",precision_score(test_labels,preds,average='macro'))
# print("f1_score_micro:",f1_score(y_true,predicts,average='micro'))
print("f1_score_macro:",f1_score(test_labels,preds,average='macro'))
# print("recall_score_micro:",recall_score(y_true,predicts,average='micro'))
print("recall_score_macro:",recall_score(test_labels,preds,average='macro'))
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
figures.plot_confusion_matrix(test_labels, preds,alphabet, "./xgb_finetune_")
