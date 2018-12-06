#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: tfrecord2csv.py
@time: 18-12-6 下午8:10
@desc:
"""
import tensorflow as tf
import pandas as pd

path_tfrecords_train="../../data/preprocessed/train_complete_16x64.tfrecord"
path_tfrecords_test="../../data/preprocessed/test_complete_16x64.tfrecord"

# 定义解析函数
def parse(serialized):
    features = {
        'recordTypes': tf.FixedLenFeature([16], tf.int64),
        'packetLength': tf.FixedLenFeature([16], tf.int64),
        'packetPayload': tf.FixedLenFeature([1024], tf.int64),
        'packetStatistic': tf.FixedLenFeature([24], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    recordTypes = parsed_example['recordTypes']
    packetLength = parsed_example['packetLength']
    packetPayload = parsed_example['packetPayload']
    packetStatistic = parsed_example['packetStatistic']
    label = parsed_example['label']
    return recordTypes, packetLength, packetPayload, packetStatistic, label

# 定义输入函数
def input_fn(filenames, train, batch_size=32, buffer_size=2048):

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parse)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    recordTypes_batch, packetLength_batch, packetPayload_batch,packetStatistic_batch, label_batch= iterator.get_next()
    packetPayload_batch=tf.cast(packetPayload_batch,tf.float32)
    packetLength_batch=tf.cast(packetLength_batch,tf.float32)
    x = {"recordTypes":recordTypes_batch,
         "packetLength":packetLength_batch,
         "packetPayload":packetPayload_batch,
         "packetStatistic":packetStatistic_batch,
         }
    y = label_batch
    return x, y

c_train=0
c_test=0
for record in tf.python_io.tf_record_iterator(path_tfrecords_test):
    c_test+=1
for record in tf.python_io.tf_record_iterator(path_tfrecords_train):
    c_train+=1

def data_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False,batch_size=c_test-1)

x,y=data_input_fn()
x_=None
y_=None
out=None
with tf.Session() as sess:
    x_,y_=sess.run([x["packetStatistic"],y])
    print(x_.shape)
    print(y_.shape)

output=pd.DataFrame(columns=["s_%d"%i for i in range(24)]+["label"])
output.to_csv("./test.csv",index=False)
df1=pd.DataFrame(x_)
print("df1 shape:",df1.shape)
df2=pd.DataFrame(y_)
print("df2 shape:",df2.shape)
df=pd.concat([df1,df2],axis=1)
print(df.shape)
df.to_csv("./test.csv",index=False,header=False,mode='a+')