#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: compare.py
@time: 18-12-4 下午8:23
@desc: 系统参数搜索
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 6))
plt.xlabel("Length of each packet")
plt.ylabel("Test Acc")
pkt_size=[8,16,32,64,80,96,112,128]
acc=[0.7542,0.842,0.9078,0.9334,0.937,0.9375,0.9376,0.9378]
plt.grid(True, ls='--')
plt.xlim(0,140)
plt.ylim(0.7,0.98)
plt.plot(pkt_size,acc,'g*:',ms=10)

for x,y in zip(pkt_size,acc):
    plt.text(x,y-0.01,"%0.3f"%y)
plt.annotate('Length:64', xy=(64, 0.925), xytext=(68, 0.88),
            arrowprops=dict(facecolor='r', shrink=0.05))
plt.savefig("./pkt_size_acc.png", format='png',bbox_inches='tight')
plt.show()
