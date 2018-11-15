#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: array2mnist.py
@time: 18-11-13 下午6:58
@desc:pandas读取csv文件--->array--->mnist
"""
import os
import pandas as pd
import numpy as np
from array import *
import config
# Load from and save to
Names = [[config.HTTPS_CONFIG["packet_payload_train_path"],'train'], [config.HTTPS_CONFIG["packet_payload_val_path"],'t10k']]

for name in Names:

    data_image = array('B')
    data_label = array('B')

    column_names=["p_%d"%i for i in range(784)]
    data=pd.read_csv(name[0])
    # 二分类： one vs all
    data["label"].apply(lambda x:0 if x==0 else 1)
    df1=data[data["label"]==0]
    print(str(df1.shape))
    df2=data[data["label"]==1].sample(n=len(df1))
    print(str(df2.shape))
    df=pd.concat([df1,df2],axis=0)
    # 数据均衡
    from sklearn.utils import shuffle
    df=shuffle(df)
    image=np.asarray(df[column_names]).reshape(1,-1)
    # label=data["label"].apply(lambda x:0 if x==0 else 1)
    label=df["label"]
    label=np.asarray(label)
    for l in label:
        data_label.append(l)
    for i in image[0]:
        data_image.append(i)

    hexval = "{0:#0{1}x}".format(len(label),6) # number of files in HEX

    # header for label array
    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))

    data_label = header + data_label
    # additional header for images array
    header.extend([0,0,0,28,0,0,0,28])

    header[3] = 3 # Changing MSB for image data (0x00000803)

    data_image = header + data_image

    output_file = open(config.HTTPS_CONFIG["ubyte"]+name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    output_file = open(config.HTTPS_CONFIG["ubyte"]+name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files
for name in Names:
    os.system('gzip '+config.HTTPS_CONFIG["ubyte"]+name[1]+'-images-idx3-ubyte')
    os.system('gzip '+config.HTTPS_CONFIG["ubyte"]+name[1]+'-labels-idx1-ubyte')