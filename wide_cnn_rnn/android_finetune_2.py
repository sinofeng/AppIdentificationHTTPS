#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: android_finetune_2.py
@time: 18-11-27 下午6:57
@desc: 使用tensor name可以指定layer计算,可以得到模型的中间计算结果
"""

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
# 数据路径
path_tfrecords_train="../../data/android_train.tfrecord"
path_tfrecords_test="../../data/android_test.tfrecord"

# 定义解析函数
def parse(serialized):
    features = {
        'recordTypes': tf.FixedLenFeature([64], tf.int64),
        'packetLength': tf.FixedLenFeature([64], tf.int64),
        'packetPayload': tf.FixedLenFeature([1024], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    recordTypes = parsed_example['recordTypes']
    packetLength = parsed_example['packetLength']
    packetPayload = parsed_example['packetPayload']
    label = parsed_example['label']
    return recordTypes, packetLength, packetPayload, label

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
    recordTypes_batch, packetLength_batch, packetPayload_batch, label_batch= iterator.get_next()
    packetPayload_batch=tf.cast(packetPayload_batch,tf.float32)
    x = {"recordTypes":recordTypes_batch,
         "packetLength":packetLength_batch,
         "packetPayload":packetPayload_batch}
    y = label_batch
    return x, y

# 训练batch
def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)
# 测试batch
def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False,batch_size=5000)

def transfer_values_cache():
    pass

x_test,y=test_input_fn()
with tf.Session() as sess:
    saver=tf.train.import_meta_graph('./Checkpoints_CNN_TF_2D/model.ckpt-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Checkpoints_CNN_TF_2D/'))
    graph = tf.get_default_graph()
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for i in tensor_name_list:
        print(i)
    x=graph.get_tensor_by_name('OneShotIterator:0')
    print(x.shape)
    x2=graph.get_tensor_by_name('IteratorGetNext:2')
    print(x2.shape)
    fc_2=graph.get_tensor_by_name('argmax:0')
    print(fc_2.shape)
    fc_2=tf.stop_gradient(fc_2)
    x_=sess.run(x_test["packetPayload"])
    print(sess.run(fc_2,feed_dict={x2:x_}))

