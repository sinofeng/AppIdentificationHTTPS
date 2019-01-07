#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: counts.py
@time: 18-12-5 下午3:49
@desc:
"""
import ast
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


packet_counts=None
with open('./counts.txt','r')as f:
    packet_counts=f.readline()

# packet_counts=packet_counts.split(":")
# for pc in packet_counts:
#     print(pc)
packet_counts=ast.literal_eval(packet_counts)
length=[]
for k,v in packet_counts.items():
    print(k,",",np.median(v),",",np.mean(v),",",len(v),",",v)
    length+=v
length=np.asarray(length)

length_count=pd.value_counts(length)
length_count_lower_16=length_count[length_count.index<17]
length_count_higher_16=length_count[length_count.index>16]
count_lower_16=sum(length_count_lower_16)
count_higher_16=sum(length_count_higher_16)

# print(length_count)
# print("avg:",np.mean(length))
# print("median",np.median(length))
# #
# fig, ax = plt.subplots(figsize=(19,9))
# sns.barplot( length_count.index,length_count.values, ax=ax)
# ax.set(xlabel= 'packet count',
#        ylabel = 'counts',
#        title = "packet numbers of a flow")
# plt.show()

labels = ['Packet Counts:>16', 'Packet Counts:<=16']
sizes = [count_higher_16,count_lower_16]
colors = ['#B1DAEF','#C0DFB0']
explode = (0.05, 0)  # explode 1st slice
plt.figure(figsize=(8, 6))
# Plot
pie=plt.pie(sizes, colors=colors,autopct='%1.1f%%',explode=explode,shadow=True, startangle=300)
plt.legend(pie[0],labels, loc='lower right', fontsize=10)
plt.axis('equal')
# plt.tight_layout()
plt.savefig("./pkts_count.png", format='png',bbox_inches='tight')
plt.show()