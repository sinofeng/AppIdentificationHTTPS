#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: https_parse_time.py
@time: 18-11-11 上午11:49
@desc:
"""
#!/usr/bin/env python
#encoding=utf-8
import sys
import hashlib
from hashlib import md5
from scapy.all import *
from scapy.all import IP,TCP
from scapy_ssl_tls.ssl_tls import *
from tcp_stream import TCPStream
import config
import argparse
import os
import threading
# 五元组确定一条流
def create_forward_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.src,pkt.sport,pkt.dst,pkt.dport,pkt.proto)
def create_reverse_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.dst,pkt.dport,pkt.src,pkt.sport,pkt.proto)
def create_flow_keys(pkt):
    return create_forward_flow_key(pkt),create_reverse_flow_key(pkt)

flows = {}

def lookup_stream(key,reverse_key):

    if key in flows.keys():
        return key,flows[key]
    elif reverse_key in flows.keys():
        return reverse_key,flows[reverse_key]
    else:
        return key,None

#HINT: appid https://code.google.com/p/appid/
def proto_name(sport,dport,use_dpi=False,payload=None):
    if dport == 80 or sport == 80:
        return "http"
    if dport == 3306 or sport == 3306:
        return "mysql"
    if dport == 22 or sport == 22:
        return "ssh"
    if dport == 443 or sport == 443:
        return "https"
    return "unknown"

def padArray(tmp,num):
    if len(tmp)>128:
        return tmp[:128]
    else:
        return tmp+[num]*(128-len(tmp))

def parse(pcap_file):
    packets=rdpcap(config.HTTPS_CONFIG["pcap_path"]+pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]
    #here we are sure ALL PACKETS ARE TCP
    for pkt in packets:
        flow_tuple = reverse_flow_tuple = key_to_search = None
        flow_tuple,reverse_flow_tuple = create_flow_keys(pkt[IP])
        flow_key,tcp_stream = lookup_stream(flow_tuple,reverse_flow_tuple)
        if tcp_stream is None:
            tcp_stream = TCPStream(pkt[IP])
        else:
            tcp_stream.add(pkt[IP])

        flows[flow_key] = tcp_stream


    with open(config.HTTPS_CONFIG["time_interval_total"]+pcap_file[:-8]+'_time_interval.csv','a')as f:
        for (flow,i) in zip(flows.values(),range(len(flows))):
            # 只有长度大于20的流才会保留
            if flow.pkt_count>=20:
                tmp=padArray(flow.inter_arrival_times,0)
                tmp=str(tmp).strip('[]')
                f.write(pcap_file[:-5]+"_"+str(i)+","+tmp+"\n")
def run(pcaps):
    for p in pcaps:
        parse(p)

if __name__ == '__main__':
    pcap_files=os.listdir(config.HTTPS_CONFIG["pcap_path"])

    record_type_names=["id"]+["r_"+str(i) for i in range(128)]+['label']
    time_interval_names=["id"]+["c_"+str(i) for i in range(128)]+['label']
    softwares=set([pcap_file[:-8] for pcap_file in pcap_files])

    for software in softwares:
        with open(config.HTTPS_CONFIG["time_interval_total"]+software+'_time_interval.csv','w+')as f:
            f.write(','.join(time_interval_names)+'\n')
    threads=[]

    for software in softwares:
        all_pcap=[i for i in pcap_files if i.find(software)>-1]
        t=threading.Thread(target=run,args=(all_pcap,))
        print ("start:%s"%software)
        t.start()
        threads.append(t)
    for k in threads:
        k.join()