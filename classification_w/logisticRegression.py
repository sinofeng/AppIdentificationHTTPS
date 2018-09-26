#/usr/bin/env python
#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import input_w
train_data,train_labels,eval_data,eval_labels=input_w.inputs()

lr=LogisticRegression()
model=lr.fit(train_data,train_labels)
print(cross_val_score(lr,train_data,train_labels,cv=5))

