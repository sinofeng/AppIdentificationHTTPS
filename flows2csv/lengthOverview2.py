#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: lengthOverview2.py
@time: 18-11-11 下午5:11
@desc:
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
length_names=["c_%s"%str(i) for i in range(128)]
path="/home/ss/workplace/experiment/data/wd_https/packet_length/total/"
length_images="/home/ss/workplace/experiment/wd_https/result/lengthOverview/"

length_data_files=os.listdir(path)

for file in length_data_files:
    df = pd.read_csv(path+file)
    # length_data=df[length_names]
    length_data=df.sample(n=10)[length_names]
    data=np.asarray(length_data)
    for t in data:
        t=t[t>0]
        plt.title(file[:-4])
        plt.plot(t)
    plt.savefig(length_images+file+".png")
    plt.show()