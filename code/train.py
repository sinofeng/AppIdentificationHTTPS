from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import os
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # raw data 32*32 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 1]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  with tf.device('/gpu:0'):
      conv1 = tf.layers.conv2d(inputs=input_layer,
                               filters=32,
                               kernel_size=[5, 5],
                               padding="same",
                               activation=tf.nn.relu)
      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 32, 32, 32]
      # Output Tensor Shape: [batch_size, 16, 16, 32]
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

      # Convolutional Layer #2
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 16, 16, 32]
      # Output Tensor Shape: [batch_size, 16, 16, 64]
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 16, 16, 64]
      # Output Tensor Shape: [batch_size, 8, 8, 64]
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 8, 8, 64]
      # Output Tensor Shape: [batch_size, 8 * 8 * 64]
      pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

      # Dense Layer
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 8 * 8 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
          inputs=dense,
          rate=0.4,
          training=mode == tf.estimator.ModeKeys.TRAIN)

      # Logits layer
      # Input Tensor Shape: [batch_size, 1024]
      # Output Tensor Shape: [batch_size, 10]
      logits = tf.layers.dense(inputs=dropout, units=3)

      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
      if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
          train_op = optimizer.minimize(
              loss=loss,
              global_step=tf.train.get_global_step())
          return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  csv_filenames=os.listdir('./processed_data/')
  # 数据集
  data_df=pd.DataFrame()
  
  for csv_filename in csv_filenames:
      df=pd.read_csv('./processed_data/'+csv_filename,header=None,sep=',')
      data_df=pd.concat([data_df,df],ignore_index=True)
      del df
  # 切分训练集和测试集
  # 训练集
  train_df=data_df.sample(frac=0.8,random_state=200)
  train_labels = np.asarray(train_df[1024],dtype=np.int32)
  train_data = np.asarray(train_df.drop([1024],axis=1),dtype=float)
  # 测试集
  eval_df=data_df.drop(train_df)
  eval_labels = np.asarray(eval_df[1024],dtype=np.int32)
  eval_data=np.asarray(eval_df.drop([1024],axis=1),dtype=float)
  del data_df,train_df,eval_df
  
  # 配置使用GPU
  run_config=tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True,device_count={'GPU':0})
      )
  # Create the Estimator
  https_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,config=run_config,model_dir="./https_classification")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  # https_classifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])
  https_classifier.train(input_fn=train_input_fn,steps=20000)
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = https_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()

