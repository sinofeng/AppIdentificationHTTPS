#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: packetcount.py
@time: 18-12-4 下午9:00
@desc:
"""
from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
import tensorflow as tf
import numpy as np
import os



def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

counts={"baiduditu":[],
           "baidutieba":[],
           "cloudmusic":[],
           "iqiyi":[],
           "jingdong":[],
           "jinritoutiao":[],
           "meituan":[],
           "qq":[],
           "qqmusic":[],
           "qqyuedu":[],
           "taobao":[],
           "weibo":[],
           "xiecheng":[],
           "zhihu":[],
           "douyin":[],
           "elema":[],
           "guotaijunan":[],
           "QQyouxiang":[],
           "tenxunxinwen":[],
           "zhifubao":[]
        }

def count_parse(pcap_file):
    packets=rdpcap("./ALL20/"+pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]
    return len(packets)
def count(pcap_files):
    nums=len(pcap_files)
    i=0
    for pcap_file in pcap_files:
        index=pcap_file.find("_")
        counts[pcap_file[:index]].append(count_parse(pcap_file))
        i+=1
        print_progress(i,nums)

pcap_files=os.listdir("./ALL20/")
count(pcap_files)
with open("./counts.txt",'w+')as f:
    f.write(str(counts))


