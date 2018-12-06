#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: drawingTestData.py
@time: 18-12-1 下午4:37
@desc:
"""
import numpy as np
import matplotlib.pyplot as plt
size = 3
mymodel = np.asarray([0.9184840801316903,0.9090443033011008,0.912511745951522])
baseline = np.asarray([0.88029597255012,0.8750779175862263,0.8768301416163474])


x = np.asarray([0,1,2])

total_width, n = 0.8, 2   # 有多少个类型，只需更改n即可
width = total_width / n
x = x - (total_width - width) / 2


plt.figure(figsize=(8, 6.5))
plt.grid(True, ls='--')
plt.ylim(0.5,1)
b1=plt.bar(x, mymodel,  width=width, label='mymodel')
b2=plt.bar(x + width, baseline, width=width, label='baseline')
plt.xticks(x+width/2,["Precision","Recall","F1"])
for b in b1+b2:
    h=b.get_height()
    plt.text(b.get_x()+b.get_width()/2,h,'%0.3f'%float(h),ha='center',va='bottom')

plt.legend()
plt.savefig("./test.png", format='png',bbox_inches='tight')
plt.show()