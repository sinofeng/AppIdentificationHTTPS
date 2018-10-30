#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: sniFilter.py
@time: 18-10-29 下午7:50
@desc: 建立SNI模式，对数据清洗
"""
SNI_PATTERNS={
    "cloudmusic":["126","163"],
    "xiecheng":[],
    "aiqiyi":["iqiyi"],
    "taobao":["taobao","tmall"],
    "weibo":["weibo","sina"],
    "guotaijunan":[],
    "shoujibaidu":["baidu"],
    "tenxunxinwen":[],
    "baiduyuedu":["baidu","yuedu"],
    "baidutieba":["baidu","tieba"],
    "jinritoutiao":[],
    "nanfangzhoumo":[],
    "qqyuedu":[],
    "zhangyue":["zhangyue","book"],
    "zhihu":[],
    "qq":["qq"],
    "weixin":[],
    "qqmusic":["qq"],
    "":[]
}

import re
import pandas as pd
import config

def matchSNI(serverName,id):
    i=id.find("_")
    patterns=SNI_PATTERNS[id[:i]]
    for pattern in patterns:
        if re.search(pattern,serverName):
            return True
    return False
all_train_data=pd.read_csv(config.HTTPS_CONFIG["all_train_path"])
all_train_data["FLAG"]=all_train_data.apply(lambda tmp : matchSNI(tmp["extension_servername_indication"],tmp['id']),axis=1)

all_train_data.loc[all_train_data["FLAG"]==True].to_csv(config.HTTPS_CONFIG["all_filtered_train_path"],header=True)

all_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])
all_val_data["FLAG"]=all_val_data.apply(lambda tmp : matchSNI(tmp["extension_servername_indication"],tmp['id']),axis=1)

all_val_data.loc[all_val_data["FLAG"]==True].to_csv(config.HTTPS_CONFIG["all_filtered_val_path"],header=True)
