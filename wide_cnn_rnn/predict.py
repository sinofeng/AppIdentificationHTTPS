#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: predict.py
@time: 19-5-30 下午4:01
@desc:
"""
import tensorflow as tf
from result import figures
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import sys
# 打印log
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
trainingConfig = tf.estimator.RunConfig(session_config=config)
# 数据路径

pkt_counts=16
pkt_size=96

path_tfrecords_train="../../data/preprocessed/train_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)
path_tfrecords_test="../../data/preprocessed/test_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)

# path_tfrecords_train="../../data/preprocessed/unb_train_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)
# path_tfrecords_test="../../data/preprocessed/unb_test_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)


choose = "approximate_attention"

train=True
steps=500
learning_rate=1e-3
# 定义解析函数
def parse(serialized):
    features = {
        'recordTypes': tf.FixedLenFeature([pkt_counts], tf.int64),
        'packetLength': tf.FixedLenFeature([pkt_counts], tf.int64),
        'packetPayload': tf.FixedLenFeature([pkt_counts*pkt_size], tf.int64),
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

# 训练batch
def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)


# 测试batch
def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False,batch_size=5000)




params = {"learning_rate": learning_rate}

export_dir="../../data/checkpoints/saved_model/1559214033"
# loading
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    graph = tf.get_default_graph()
    print("success")

from tensorflow.contrib import predictor
# 可以在这里指定输出的具体层数!在网络中设置一个键值,在进行预测的时候直接取值进行预测!
# 如:需要指定拿到概率值!
predict_fn=predictor.from_saved_model(export_dir)
predicts=predict_fn({"packetPayload":[[0]*1536],"recordTypes":[[9]*16]})

print(predicts['output'])