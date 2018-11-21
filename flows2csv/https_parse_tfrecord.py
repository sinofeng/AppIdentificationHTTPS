#!/usr/bin/env python
#coding=utf-8
from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
import tensorflow as tf
import numpy
import os
import config

writer=tf.python_io.TFRecordWriter("data.tfrecords")
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

    i=pcap_file.find("_")
    label=softwares[pacp_file[:i]]
    return recordTypes,packetLength,packetPayload,label
start=time.time()

pcap_files=os.listdir(config.HTTPS_CONFIG[""])

for pacp_file in pcap_files:
    recordTypes, packetLength, packetPayload,label=packet_parse(config.HTTPS_CONFIG+pacp_file)
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