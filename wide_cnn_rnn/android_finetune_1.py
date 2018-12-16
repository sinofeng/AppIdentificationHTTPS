#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: android_finetune_1.py
@time: 18-11-27 下午6:50
@desc: 加载模型的权重,从头开始训练,利用字典的方式加载原始模型的参数
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
# 打印log
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
trainingConfig = tf.estimator.RunConfig(session_config=config)
# 数据路径

pkt_counts=int(sys.argv[1])
pkt_size=int(sys.argv[2])

# path_tfrecords_train="../../data/preprocessed/train_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)
# path_tfrecords_test="../../data/preprocessed/test_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)

path_tfrecords_train="../../data/preprocessed/unb_train_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)
path_tfrecords_test="../../data/preprocessed/unb_test_complete_%dx%d.tfrecord"%(pkt_counts,pkt_size)


# choose = "model_cnn1d"
# choose = "model_cnn1d_2"
# choose = "model_cnn1d_rnn"
# choose = "model_dnn"
# choose = "model_cnn1d_rnn"
choose = "model_cnn1d_cnn1d_rnn"
# choose = "model_cnn_rnn_dnn"

train=True

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
    # return input_fn(filenames=path_tfrecords_test, train=False,batch_size=5000)
    return input_fn(filenames=path_tfrecords_test, train=False,batch_size=500)

# 定义模型
def model_fn(features, labels, mode, params):

    x1 = features["packetPayload"]
    x1 = tf.layers.batch_normalization(inputs=x1)
    net1 = tf.reshape(x1, [-1, pkt_counts*pkt_size, 1])

    # First convolutional layer.
    net1 = tf.layers.conv1d(inputs=net1, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net1 = tf.layers.max_pooling1d(inputs=net1, pool_size=3,strides=1)
    # net1 = tf.layers.max_pooling1d(inputs=net1, pool_size=2, strides=2)

    net1 = tf.layers.conv1d(inputs=net1, name='layer_conv2',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net1 = tf.layers.max_pooling1d(inputs=net1, pool_size=3,strides=1)
    # net1 = tf.layers.max_pooling1d(inputs=net1, pool_size=2, strides=2)
    net1 = tf.contrib.layers.flatten(net1)
    net1 = tf.layers.dense(inputs=net1, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    x2 = features["recordTypes"]
    net2 = tf.reshape(x2,[-1,pkt_counts])
    # Embedding
    word_embeddings = tf.get_variable("word_embeddings",[257, 32])


    net2 = tf.nn.embedding_lookup(word_embeddings, net2)
    # Rnn
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(64)
    output, states = tf.nn.dynamic_rnn(rnn_cell, net2, dtype=tf.float32)
    net2 = tf.layers.dense(inputs=output[:,-1,:], name='layer_rnn_fc_1',
                          units=128, activation=tf.nn.relu)
    x3 = features["packetStatistic"]
    net3 = tf.layers.batch_normalization(inputs=x3)
    net3 = tf.layers.dense(inputs=net3,name="layer_dnn_1",units=128,activation=tf.nn.relu)
    net3 = tf.layers.dropout(inputs=net3,rate=0.2)
    net3 = tf.layers.dense(inputs=net3,name="layer_dnn_2",units=128,activation=tf.nn.relu)
    net3 = tf.layers.dropout(inputs=net3,rate=0.2)
    x4 = features["packetLength"]
    x4 = tf.layers.batch_normalization(inputs=x4)
    net4 = tf.reshape(x4, [-1, pkt_counts, 1])

    # First convolutional layer.
    net4 = tf.layers.conv1d(inputs=net4, name='layer_length_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net4 = tf.layers.max_pooling1d(inputs=net4, pool_size=2, strides=2)

    net4 = tf.layers.conv1d(inputs=net4, name='layer_length_conv2',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net4 = tf.layers.max_pooling1d(inputs=net4, pool_size=2, strides=2)

    net4 = tf.contrib.layers.flatten(net4)
    net4 = tf.layers.dense(inputs=net4, name='layer_length_fc1',
                          units=128, activation=tf.nn.relu)
    if choose == "model_cnn1d":
        net = tf.layers.dense(inputs=net1, name='layer_combine_fc_cnn1d', units=128, activation=tf.nn.relu)
    if choose == "model_cnn1d_2":
        net = tf.concat([net1,net4],1)
    if choose == "model_cnn1d_rnn":
        net = tf.concat([net1,net2],1)
    if choose == "model_dnn":
        net=net3
    if choose == "model_cnn1d_cnn1d_rnn":
        net = tf.concat([net1,net2,net4],1)
    if choose == "model_cnn_rnn_dnn":
        net = tf.concat([net1,net2,net3,net4],1)

    net = tf.layers.dense(inputs=net, name='layer_combine_fc_x',units=128,activation=tf.nn.relu)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prob = 1.0
    else:
        prob =0.8
    net_combine = tf.layers.dropout(inputs=net,rate=prob)
    net = tf.layers.dense(inputs=net, name='layer_combine_fc_y',units=20)
    # net = tf.layers.dense(inputs=net, name='layer_combine_fc_y',units=14)


    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions={
            'layer_combine':net_combine,
            'y_pred_cls':y_pred_cls
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        accuracy = tf.metrics.accuracy(labels, y_pred_cls)
        tf.summary.scalar('accuracy',accuracy[1])
        metrics = \
            {
                "accuracy": accuracy
            }
        # 早停机制
        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook()
        logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                   "accuracy": accuracy[1]}, every_n_iter=10)
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_hooks=[logging_hook]
        )

    return spec


params = {"learning_rate": 1e-4}

model = tf.estimator.Estimator(model_fn=model_fn,
                               config=trainingConfig,
                               params=params,
                               model_dir="../../data/checkpoints/checkpoints_%dx%d_2" % (pkt_counts, pkt_size) + choose)

c_train=0
c_test=0
for record in tf.python_io.tf_record_iterator(path_tfrecords_test):
    c_test+=1
for record in tf.python_io.tf_record_iterator(path_tfrecords_train):
    c_train+=1

def data_input_fn():
    # return input_fn(filenames=path_tfrecords_train, train=False, batch_size=c_train - 1)
    return input_fn(filenames=path_tfrecords_test, train=False,batch_size=c_test-1)



# 模型预测
predicts=model.predict(input_fn=data_input_fn,predict_keys=["layer_combine"])
predicts=[p for p in predicts]

#
_,y=data_input_fn()
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
y_true=sess.run(y)

x_name=["x_%s"%str(i) for i in range(128)]
x_name.append("label")

output=pd.DataFrame(columns=x_name)
output.to_csv("../../data/finetune_test.csv",index=False)

with open("../../data/finetune_test.csv",'a')as f:
    for (x,y) in zip(predicts,y_true):
        x_tmp=x['layer_combine']
        x_tmp=list(x_tmp)
        x_tmp.append(y)
        f.write(str(x_tmp).strip('[]')+'\n')