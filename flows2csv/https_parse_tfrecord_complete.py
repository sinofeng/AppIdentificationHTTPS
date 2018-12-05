#!/usr/bin/env python
#coding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: https_parse.py
@time: 18-11-12 下午7:50
@desc: 解析包,将ascii写入到tfrecord中.特征包括:统计特征,payload,content type,packet长度
可以变化的包括:payload的长度,packet的数目
"""

from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
import tensorflow as tf
import numpy as np
import os
import sys

pkt_counts = int(sys.argv[1])
pkt_size = int(sys.argv[2])
result_path = "%dx%d"%(pkt_counts,pkt_size)



def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def wrap_array_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
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
           "zhihu":13,
           "douyin":14,
           "elema":15,
           "guotaijunan":16,
           "QQyouxiang":17,
           "tenxunxinwen":18,
           "zhifubao":19
           }

# softwares={"AIM":0,
#            "email":1,
#            "facebookchat":2,
#            "gmailchat":3,
#            "hangoutsaudio":4,
#            "hangoutschat":5,
#            "icqchat":6,
#            "netflix":7,
#            "skypechat":8,
#            "skypefile":9,
#            "spotify":10,
#            "vimeo":11,
#            "youtube":12,
#            "youtubeHTML5":13}

######################################################################################
def statistics(nums):
    if len(nums)>0:
        return [np.mean(nums),np.max(nums),np.min(nums),np.median(nums),np.var(nums)]
    else:
        return [0,0,0,0,0]

def getRecordType(pkt):
	try:
		recordtype=pkt["TLS Record"].content_type
	except:
		try:
			recordtype=pkt["SSLv2 Record"].content_type
		except:
			recordtype=256
	return recordtype
def session_id_len(pkt):
    fields=["TLS Client Hello","TLS Server Hello"]
    for field in fields:
        try:
            tmp= pkt[field].session_id_length
        except:
            tmp= 0
    return 0

def client_extensions_len(pkt):
    tmp=0
    if str(pkt.sprintf).find("TLSClientHello")>-1:
        try:
            tmp= pkt["TLS Client Hello"].extensions_length
        except:
            tmp=0
    if not tmp:
        tmp=0
    return tmp
def session_ticket_lifetime(pkt):
    if str(pkt.sprintf).find("TLSSessionTicket")>-1:
        return pkt["TLS Session Ticket"].lifetime
    return 0

def ciphers(pkt):
    if str(pkt.sprintf).find("TLSClientHello")>-1 :
        return pkt["TLS Client Hello"].cipher_suites
    else:
        return []

def cipher(pkt):
    if str(pkt.sprintf).find("TLSServerHello")>-1:
        try:
            return [pkt["TLS Server Hello"].cipher_suite]
        except:
            return []
    else:
        return []

def getPacketStatistic(packets):
    output=[]
    # 1 packet counts
    count=len(packets)
    output.append(count)
    # 5 ttl
    ttl=[pkt["IP"].ttl for pkt in packets]
    output+=statistics(ttl)
    # 5 packet length
    length=[pkt.len for pkt in packets]
    output+=statistics(length)
    # 5 packet window
    window=[pkt["TCP"].window for pkt in packets]
    output+=statistics(window)
    # 1 seesion id length
    session_id_length=[session_id_len(pkt) for pkt in packets]
    if len(session_id_length)>1:
        output.append(max(session_id_length))
    else:
        output.append(0)

    # 5 client extensions length
    client_extensions_length=[client_extensions_len(pkt) for pkt in packets]
    output+=statistics(client_extensions_length)
    # 2
    client_ciphers=set([])
    server_cipher=set([])
    for pkt in packets:
        client_ciphers=client_ciphers|set(ciphers(pkt))
        server_cipher=server_cipher|set(cipher(pkt))
    output.append(len(client_ciphers))
    if len(server_cipher)==0:
        output.append(-1)
    else:
        output.append(list(server_cipher)[0])
    return output
#######################################################################################

def packet_parse(pcap_file):
    '''
    解析pcap文件
    :param pcap_file:
    :return:content type,packet长度,payload,label
    '''
    packets = rdpcap("../../data/wd_https/noVPN/"+pcap_file)
    # packets = rdpcap("../../data/wd_https/noVPN/" + pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]
    recordTypes=[getRecordType(pkt) for pkt in packets]
    recordTypes=padArray(recordTypes,256,64)

    packetLength=[pkt.len for pkt in packets]
    packetLength=padArray(packetLength,0,64)

    payloads=""
    for pkt in packets[:pkt_counts]:
        payloads+=str(pkt.payload)[:pkt_size]

    packetPayload=[ord(c) for c in payloads]
    packetPayload=padArray(packetPayload,-1,pkt_counts*pkt_size)
    packetStatistic=getPacketStatistic(packets)
    i=pcap_file.find("_")
    label=softwares[pcap_file[:i]]
    return recordTypes,packetLength,packetPayload,packetStatistic,label



pcap_files=os.listdir("../../data/wd_https/noVPN/")
# pcap_files=os.listdir("../../data/wd_https/noVPN/")

np.random.shuffle(pcap_files)

train_files=pcap_files[5000:]
test_files=pcap_files[:5000]

def convert(pcap_files, out_path):
    print("Converting: " + out_path)
    num = len(pcap_files)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        i=0
        for pacp_file in pcap_files:
            print_progress(i, num)
            recordTypes, packetLength, packetPayload, packetStatistic, label = packet_parse(pacp_file)
            example = tf.train.Example(features=tf.train.Features(
                feature={'recordTypes': wrap_array(recordTypes),
                         'packetLength': wrap_array(packetLength),
                         'packetPayload': wrap_array(packetPayload),
                         'packetStatistic': wrap_array_float(packetStatistic),
                         'label': wrap_int64(label)
                         }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            i = i + 1

start=time.time()

convert(train_files,"../../data/wd_https/train_complete_%s.tfrecord"%result_path)
convert(test_files,"../../data/wd_https/test_complete_%s.tfrecord"%result_path)

end=time.time()
print ("time cost:",end-start)