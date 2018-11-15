#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: ubyteOverview.py
@time: 18-11-14 上午10:08
@desc:
"""

from tensorflow.examples.tutorials.mnist import input_data
import config
import numpy as np
train,val,test = input_data.read_data_sets(config.HTTPS_CONFIG["ubyte"])
# train,val,test = input_data.read_data_sets(config.HTTPS_CONFIG["ubyte"],one_hot=True)
print("*"*50)
train_x=np.asarray(train.images)
print(train_x.shape)
train_y=np.asarray(train.labels)
print(train_y.shape)
print(len(train_y[train_y==0]))
print(len(train_y[train_y==1]))

# print(set(train_y))
print("*"*50)
val_x=np.asarray(val.images)
print(val_x.shape)
val_y=np.asarray(val.labels)
print(val_y.shape)
print(len(val_y[val_y==0]))
print(len(val_y[val_y==1]))
print("*"*50)
test_x=np.asarray(test.images)
print(test_x.shape)
test_y=np.asarray(test.labels)
print(test_y.shape)
print(len(test_y[test_y==0]))
print(len(test_y[test_y==1]))
# print(set(test_y))
print("*"*50)