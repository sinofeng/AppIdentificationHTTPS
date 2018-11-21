#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: lengthOverview.py
@time: 18-11-11 上午9:54
@desc:
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config

packet_length_names=["Seq_"+str(i) for i in range(128)]

path="/home/ss/workplace/experiment/data/wd_https/https_train_eval/packet_length.csv"
len_data=pd.read_csv(path)

def getLenMedian(s):
    nums=np.asarray(s)
    nums=nums[nums>0]
    mid=np.median(nums)
    return mid
def getLenMean(s):
    nums=np.asarray(s)
    nums=nums[nums>0]
    avg=np.mean(nums)
    return avg
len_data["mid"]=len_data.apply(lambda tmp:getLenMedian(tmp[packet_length_names]),axis=1)
len_data["avg"]=len_data.apply(lambda tmp:getLenMean(tmp[packet_length_names]),axis=1)

for label in len_data["label"].unique():
    plt.figure(figsize=(10,4))
    sns.distplot(len_data[len_data["label"]==label]["mid"])
    # plt.show()
    plt.savefig(config.HTTPS_CONFIG["result"]+"median_"+str(label)+".png")

for label in len_data["label"].unique():
    plt.figure(figsize=(10,4))
    sns.distplot(len_data[len_data["label"]==label]["avg"])
    # plt.show()
    plt.savefig(config.HTTPS_CONFIG["result"]+"mean_"+str(label)+".png")