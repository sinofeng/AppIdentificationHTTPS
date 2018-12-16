#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: loss_acc.py
@time: 18-11-26 下午6:50
@desc:
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
loss_cnn1d=pd.read_csv("../../data/loss_cnn_1D.csv")
loss_cnn2d=pd.read_csv("../../data/loss_cnn_2d.csv")
loss_cnn1d_rnn=pd.read_csv("../../data/loss_cnn1d_rnn.csv")
loss_cnn1d_cnn1d_rnn=pd.read_csv("../../data/loss_cnn1d_cnn1d_rnn.csv")
loss_cnn1d_cnn1d=pd.read_csv("../../data/loss_cnn1d_cnn1d.csv")
x=loss_cnn1d["Step"][1:]
y1=loss_cnn1d["Value"][1:]
y2=loss_cnn2d["Value"][1:]
y3=loss_cnn1d_rnn["Value"][1:]
y4=loss_cnn1d_cnn1d_rnn["Value"][1:]
y5=loss_cnn1d_cnn1d["Value"][1:]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 12))

ax=plt.subplot(211)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.plot(x, y1, 'r-', mec='k', label='Loss cnn_1d', lw=1)
plt.plot(x, y2, 'g-', mec='k', label='Loss cnn_2d', lw=1)
plt.plot(x, y3, 'b-',mec='k', label='Loss CRC', lw=1)
# plt.plot(x, y4, color='olive' ,linestyle='-', mec='k', label='Loss cnn1d_cnn1d_rnn', lw=1)
# plt.plot(x, y5, color='orange' ,linestyle='-', mec='k', label='Loss cnn1d_cnn1d', lw=1)
# plt.plot(x, boost, 'm--',mec='k', label='Adaboost Loss',lw=2)
plt.grid(True, ls='--')
plt.legend(loc='upper right')
plt.title('(1) Loss over steps',y=-0.18)
# plt.savefig('../result/loss.png')
# plt.show()


acc_cnn_2d=pd.read_csv("../../data/acc_cnn_2d.csv")
acc_cnn_1d=pd.read_csv("../../data/acc_cnn_1d.csv")
acc_cnn1d_rnn=pd.read_csv("../../data/acc_cnn1d_rnn.csv")
acc_cnn1d_cnn1d_rnn=pd.read_csv("../../data/acc_cnn1d_cnn1d_rnn.csv")
acc_cnn1d_cnn1d_rnn=pd.read_csv("../../data/acc_cnn1d_cnn1d_rnn.csv")
acc_cnn1d_cnn1d=pd.read_csv("../../data/acc_cnn1d_cnn1d.csv")

x_acc_cnn_2d=acc_cnn_2d["Step"]
y_acc_cnn_2d=acc_cnn_2d["Value"]
y_acc_cnn_1d=acc_cnn_1d["Value"]
y_acc_cnn1d_rnn=acc_cnn1d_rnn["Value"]
y_acc_cnn1d_cnn1d_rnn=acc_cnn1d_cnn1d_rnn["Value"]
y_acc_cnn1d_cnn1d=acc_cnn1d_cnn1d["Value"]
plt.subplot(212)
# plt.figure(figsize=(8, 5))
plt.xlabel("Steps")
plt.ylabel("Acc")
plt.plot(x_acc_cnn_2d, y_acc_cnn_2d, 'r-', mec='k', label='Acc CNN_2D', lw=1)
plt.plot(x_acc_cnn_2d,y_acc_cnn_1d, 'g-', mec='k', label='Acc CNN_1D', lw=1)
plt.plot(x_acc_cnn_2d, y_acc_cnn1d_rnn, 'b-',mec='k', label='Acc CRC', lw=1)
# plt.plot(x_acc_cnn_2d, y_acc_cnn1d_cnn1d_rnn,color='olive' ,linestyle='-',mec='k', label='Acc CNN1D_CNN1D_RNN', lw=1)
# plt.plot(x_acc_cnn_2d, y_acc_cnn1d_cnn1d,color='orange' ,linestyle='-',mec='k', label='Acc CNN1D_CNN1D', lw=1)


# plt.plot(x, boost, 'm--',mec='k', label='Adaboost Loss',lw=2)
plt.grid(True, ls='--')
plt.legend(loc='lower right')
plt.title('(2) Accuracy over steps',y=-0.18)
plt.subplots_adjust(hspace=0.25)
plt.savefig('../result/acc_loss.png',bbox_inches='tight')
plt.show()


