#!/usr/bin/env python
#coding=utf-8
from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
import tensorflow as tf
import numpy as np
import os

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

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

softwares={"baiduditu":0,
           "baidutieba":1,
           "cloudmusic":2,
           "iqiyi":3,
           "jingdong":4,
           "jinritoutiao":5,
           "meituan":6,
           "qq":7,
           "qqmusic":8,
           "qqyuedu":9,
           "taobao":10,
           "weibo":11,
           "xiecheng":12,
           "zhihu":13}

def packet_parse(pcap_file):
    packets=rdpcap("./ALL"+pcap_file)
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

    i=pcap_file.find("_")
    label=softwares[pcap_file[:i]]
    return recordTypes,packetLength,packetPayload,label



pcap_files=os.listdir("./ALL/")
np.random.shuffle(pcap_files)

train_files=pcap_files[5000:]
test_files=pcap_files[:5000]

def convert(pcap_files, out_path):
    print("Converting: " + out_path)
    num = len(pcap_files)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        i=0
        for pacp_file in pcap_files:
            print_progress(num, i)
            recordTypes, packetLength, packetPayload, label = packet_parse(pacp_file)
            example = tf.train.Example(features=tf.train.Features(
                feature={'recordTypes': wrap_array(recordTypes),
                         'packetLength': wrap_array(packetLength),
                         'packetPayload': wrap_array(packetPayload),
                         'label': wrap_int64(label)
                         }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            i = i + 1


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
    x = {"recordTypes":recordTypes_batch,
         "packetLength":packetLength_batch,
         "packetPayload":packetPayload_batch}
    y = tf.one_hot(label_batch)

    return x, y
#
# start=time.time()
# convert(train_files,"train.tfrecord")
# convert(test_files,"test.tfrecord")
# end=time.time()
# print ("time cost:",end-start)