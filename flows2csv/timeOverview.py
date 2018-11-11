#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: timeOverview.py
@time: 18-11-11 下午1:09
@desc:
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
time_names=["c_%s"%str(i) for i in range(128)]
path="/home/ss/workplace/experiment/data/wd_https/time_interval/total/"
time_images="/home/ss/workplace/experiment/wd_https/result/timeOverview/"

time_data_files=os.listdir(path)

for file in time_data_files:
    df = pd.read_csv(path+file)
    time_data=df[time_names]
    # time_data=df.sample(n=50)[time_names]
    data=np.asarray(time_data)
    for t in data:
        t=t[t>0]
        plt.title(file[:-4])
        plt.plot(t)
    plt.savefig(time_images+file+".png")
    plt.show()