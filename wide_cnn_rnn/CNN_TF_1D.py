#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from result import figures
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score

# 打印log
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
trainingConfig = tf.estimator.RunConfig(session_config=config)
# 数据路径
path_tfrecords_train="../../data/preprocessed/train_complete_784.tfrecord"
path_tfrecords_test="../../data/preprocessed/test_complete_784.tfrecord"

# 定义解析函数
def parse(serialized):
    features = {
        'recordTypes': tf.FixedLenFeature([64], tf.int64),
        'packetLength': tf.FixedLenFeature([64], tf.int64),
        'packetPayload': tf.FixedLenFeature([784], tf.int64),
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

# 定义模型
def model_fn(features, labels, mode, params):

    x = features["packetPayload"]
    net = tf.reshape(x, [-1,784, 1])

    # First convolutional layer.
    net = tf.layers.conv1d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv1d(inputs=net, name='layer_conv2',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    net = tf.layers.dense(inputs=net, name='layer_fc_2',units=20)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        accuracy=tf.metrics.accuracy(labels, y_pred_cls)
        # 在tensorboard中可以导出loss和accuracy的csv格式文件
        tf.summary.scalar('accuracy', accuracy[1])
        metrics = \
            {
                "accuracy": accuracy
            }
        logging_hook = tf.train.LoggingTensorHook({"accuracy": accuracy[1],"loss":loss},every_n_iter=20)
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
                               model_dir="./Checkpoints_CNN_TF_1D_20/")

# 训练模型
model.train(input_fn=train_input_fn, steps=40000)

# 评估模型
result = model.evaluate (input_fn=test_input_fn)
print(result)

# 模型预测
predicts=model.predict(input_fn=test_input_fn)
print(predicts)
predicts=[p for p in predicts]
print(predicts)

_,y=test_input_fn()
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
y_true=sess.run(y)

alphabet=softwares=["baiduditu","baidutieba","cloudmusic","iqiyi","jingdong","jinritoutiao","meituan","qq","qqmusic","qqyuedu","taobao","weibo","xiecheng","zhihu","douyin","elema","guotaijunan","QQyouxiang","tenxunxinwen","zhifubao"]
figures.plot_confusion_matrix(y_true, predicts,alphabet, "./")

print("accuracy_score:",accuracy_score(y_true,predicts))
print("precision_score:",precision_score(y_true,predicts))
# print("f1_score_micro:",f1_score(y_true,predicts,average='micro'))
print("f1_score_macro:",f1_score(y_true,predicts,average='macro'))
# print("recall_score_micro:",recall_score(y_true,predicts,average='micro'))
print("recall_score_macro:",recall_score(y_true,predicts,average='macro'))

