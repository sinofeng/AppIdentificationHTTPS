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
    patterns=SNI_PATTERNS[id]
    for pattern in patterns:
        if re.search(pattern,serverName):
            return True
    return False
combinedData=pd.read_csv(config.HTTPS_CONFIG["combined_data"])
combinedData["FLAG"]=matchSNI(combinedData["sni"],combinedData['id'])
combinedData[combinedData.iloc[combinedData["FLAG"]==True]].to_csv(config.HTTPS_CONFIG["filtered_data"],header=True)
