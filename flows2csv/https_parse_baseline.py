#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: https_parse_baseline.py
@time: 18-12-1 下午2:38
@desc: 复现中科大模型数据
"""

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

# softwares={"baiduditu":0,
#            "baidutieba":1,
#            "cloudmusic":2,
#            "iqiyi":3,
#            "jingdong":4,
#            "jinritoutiao":5,
#            "meituan":6,
#            "qq":7,
#            "qqmusic":8,
#            "qqyuedu":9,
#            "taobao":10,
#            "weibo":11,
#            "xiecheng":12,
#            "zhihu":13}

softwares={"AIM":0,
           "email":1,
           "facebookchat":2,
           "gmailchat":3,
           "hangoutsaudio":4,
           "hangoutschat":5,
           "icqchat":6,
           "netflix":7,
           "skypechat":8,
           "skypefile":9,
           "spotify":10,
           "vimeo":11,
           "youtube":12,
           "youtubeHTML5":13}


def packet_parse(pcap_file):
    '''
    解析pcap文件
    :param pcap_file:
    :return:content type,packet长度,payload,label
    '''
    packets=rdpcap("../../data/wd_https/noVPN/"+pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]

    payloads=""
    for pkt in packets[:20]:
        payloads+=str(pkt.payload)
    packetPayload=[ord(c) for c in payloads]
    packetPayload=padArray(packetPayload,-1,784)

    i=pcap_file.find("_")
    label=softwares[pcap_file[:i]]
    return packetPayload,label



pcap_files=os.listdir("../../data/wd_https/noVPN/")
np.random.shuffle(pcap_files)

train_files=pcap_files[500:]
test_files=pcap_files[:500]

def convert(pcap_files, out_path):
    print("Converting: " + out_path)
    num = len(pcap_files)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        i=0
        for pacp_file in pcap_files:
            print_progress(i, num)
            recordTypes, packetLength, packetPayload, label = packet_parse(pacp_file)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                         'packetPayload': wrap_array(packetPayload),
                         'label': wrap_int64(label)
                         }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            i = i + 1


def parse(serialized):
    features = {
        'packetPayload': tf.FixedLenFeature([784], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    packetPayload = parsed_example['packetPayload']
    label = parsed_example['label']
    return packetPayload, label


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
    x = {"packetPayload":packetPayload_batch}
    y = tf.one_hot(label_batch)

    return x, y

start=time.time()
convert(train_files,"../../data/wd_https/no_vpn_train.tfrecord")
convert(test_files,"../../data/wd_https/no_vpn_test.tfrecord")
end=time.time()
print ("time cost:",end-start)