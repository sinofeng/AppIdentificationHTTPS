import tensorflow as tf

filename_queue = tf.train.string_input_producer(['../../data/preprocessed/train_complete_16x80.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'recordTypes': tf.FixedLenFeature([16], tf.int64),
                                       'packetLength': tf.FixedLenFeature([16], tf.int64),
                                       'packetPayload': tf.FixedLenFeature([1280], tf.int64),
                                       'packetStatistic':tf.FixedLenFeature([24],tf.float32),
                                       'label': tf.FixedLenFeature([],tf.int64)
                                   }
                                  )

r_out = features['recordTypes']
l_out = features['packetLength']
p_out = features['packetPayload']
m_out = features['packetStatistic']
s_out = features['label']


print (r_out)
print (l_out)
print (p_out)
print (m_out)
print (s_out)

a_batch, b_batch, c_batch, d_batch,e_batch = tf.train.shuffle_batch([r_out, l_out, p_out, m_out,s_out], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
a_val, b_val, c_val, d_val, e_val = sess.run([a_batch, b_batch, c_batch, d_batch, e_batch])
print("="*20)
print ('first batch:')
print ('a_val:',a_val.tolist(),len(a_val[0]))
print ('b_val:',b_val.tolist(),len(b_val[0]))
print ('c_val:',c_val.tolist(),len(c_val[0]))
print ('d_val:',d_val.tolist(),len(d_val[0]))
print ('e_val:',e_val)
a_val, b_val, c_val ,d_val= sess.run([a_batch, b_batch, c_batch,d_batch])
print ('second batch:')
print ('a_val:',a_val.tolist(),len(a_val[0]))
print ('b_val:',b_val.tolist(),len(b_val[0]))
print ('c_val:',c_val.tolist(),len(c_val[0]))
print ('d_val:',d_val.tolist(),len(d_val[0]))
print ('e_val:',e_val)

