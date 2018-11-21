#!/usr/bin/env python
#coding=utf-8
from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
import tensorflow as tf
import numpy
writer=tf.python_io.TFRecordWriter("train.tfrecords")

def getRecordType(pkt):
	try:
		recordtype=pkt["TLS Record"].content_type
	except:
		try:
			recordtype=pkt["SSLv2 Record"].content_type
		except:
			recordtype=256
	return recordtype

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def padArray(array_,num,length):
    if len(array_)>length:
        return array_[:length]
    else:
        return array_+[num]*(length-len(array_))
softwares={"test_01.pcap":0,"test_02.pcap":1}
def packet_parse(pcap_file):
    packets=rdpcap(pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]
    recordTypes=[getRecordType(pkt) for pkt in packets]
    recordTypes=padArray(recordTypes,256,64)

    packetLength=[pkt.len for pkt in packets]
    packetLength=padArray(packetLength,0,64)

    payloads=""
    for pkt in packets[:20]:
        payloads+=str(pkt.payload)[:64]
    packetPayload=[ord(c) for c in payloads]
    packetPayload=padArray(packetPayload,-1,1024)

    label=softwares[pacp_file]
    return recordTypes,packetLength,packetPayload,label
start=time.time()
pcap_files=["test_01.pcap","test_02.pcap"]
for pacp_file in pcap_files:
    recordTypes, packetLength, packetPayload,label=packet_parse(pacp_file)
    example = tf.train.Example(features=tf.train.Features(
        feature={'recordTypes': wrap_array(recordTypes),
                 'packetLength': wrap_array(packetLength),
                 'packetPayload':wrap_array(packetPayload),
                 'label':wrap_int64(label)
                 }))
    serialized = example.SerializeToString()
    writer.write(serialized)
    print ('writer', pacp_file, 'done')

writer.close()
end=time.time()
print ("time cost:",end-start)
# output file name string to a queue
filename_queue = tf.train.string_input_producer(['train.tfrecords'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'recordTypes': tf.FixedLenFeature([64], tf.int64),
                                       'packetLength': tf.FixedLenFeature([64], tf.int64),
                                       'packetPayload': tf.FixedLenFeature([1024], tf.int64),
                                       'label': tf.FixedLenFeature([],tf.int64)
                                   }
                                  )

r_out = features['recordTypes']
l_out = features['packetLength']
p_out = features['packetPayload']
s_out = features['label']


print (r_out)
print (l_out)
print (p_out)
print (s_out)

a_batch, b_batch, c_batch, d_batch = tf.train.shuffle_batch([r_out, l_out, p_out, s_out], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
a_val, b_val, c_val, d_val = sess.run([a_batch, b_batch, c_batch, d_batch])
print("="*20)
print ('first batch:')
print ('a_val:',a_val,len(a_val[0]))
print ('b_val:',b_val,len(b_val[0]))
print ('c_val:',c_val,len(c_val[0]))
print ('d_val:',d_val)
a_val, b_val, c_val ,d_val= sess.run([a_batch, b_batch, c_batch,d_batch])
print ('second batch:')
print ('a_val:',a_val,len(a_val[0]))
print ('b_val:',b_val,len(b_val[0]))
print ('c_val:',c_val,len(c_val[0]))
print ('d_val:',d_val)
