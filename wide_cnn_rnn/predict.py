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

export_dir="../../data/checkpoints/saved_model/1559302829"
# loading
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    graph = tf.get_default_graph()

    for op in tf.get_default_graph().get_operations():
        print(op.name)

    print("success")

from tensorflow.contrib import predictor
# 可以在这里指定输出的具体层数!在网络中设置一个键值,在进行预测的时候直接取值进行预测!
# 如:需要指定拿到概率值!
predict_fn=predictor.from_saved_model(export_dir)

a=[range(16)]

predicts=predict_fn({"packetPayload":[[0.0]*1536],"recordTypes":a})

print(predicts['output'])