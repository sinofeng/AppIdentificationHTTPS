#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: https_parse.py
@time: 18-10-29 下午7:50
@desc: 使用多线程来解包，每一种软件处理开启一个线程
"""
import sys
sys.path.append("/home/tianmao/wd_https/")
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

# 流的方向
def create_forward_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.src,pkt.sport,pkt.dst,pkt.dport,pkt.proto)
def create_reverse_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.dst,pkt.dport,pkt.src,pkt.sport,pkt.proto)
def create_flow_keys(pkt):
    return create_forward_flow_key(pkt),create_reverse_flow_key(pkt)

flows = {}

attrs = ['protocol_name',
         'src',
         'sport',
         'dst',
         'dport',
         'proto',
         'extension_servername_indication',
         'push_flag_ratio',
         'average_len',
         'average_payload_len',
         'pkt_count',
         'flow_average_inter_arrival_time',
         'kolmogorov',
         'shannon',
         'max_len',
         'min_len',
         'std_len',
         #'len_extension_signature_algorithms',
         'len_cipher_suites',
         'avrg_tcp_window',
         'max_tcp_window',
         'min_tcp_window',
         'var_tcp_window',
         'session_id_length',
         'avrg_ip_ttl',
         'max_ip_ttl',
         'min_ip_ttl'
         ]

def lookup_stream(key,reverse_key):
    if key in flows.keys():
        return key,flows[key]
    elif reverse_key in flows.keys():
        return reverse_key,flows[reverse_key]
    else:
        return key,None

def lookup_stream_single(key):
    if key in flows.keys():
        return key,flows[key]
    else:
        return key,None

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

def padArray(tmp,l,num):
    if len(tmp)>l:
        return tmp[:l]
    else:
        return tmp+[num]*(l-len(tmp))
#  问题：如何对流进行重组（序列问题）？

def parse(pcap_file):
    packets=rdpcap(config.HTTPS_CONFIG["pcap_path"]+pcap_file)
    packets = [ pkt for pkt in packets if IP in pkt for p in pkt if TCP in p ]
    #here we are sure ALL PACKETS ARE TCP
    for pkt in packets:
        flow_tuple = reverse_flow_tuple = key_to_search = None
        flow_tuple,reverse_flow_tuple = create_flow_keys(pkt[IP])
        #混合流
        # flow_key,tcp_stream = lookup_stream(flow_tuple,reverse_flow_tuple)
        #单向流
        flow_key,tcp_stream=lookup_stream_single(flow_tuple)
        if tcp_stream is None:
            tcp_stream = TCPStream(pkt[IP])
        else:
            tcp_stream.add(pkt[IP])

        flows[flow_key] = tcp_stream
    a=config.HTTPS_CONFIG["total_path"]+pcap_file[:-8]+'.csv'
    b=config.HTTPS_CONFIG["record_type_total"]+pcap_file[:-8]+'_record_type.csv'
    c=config.HTTPS_CONFIG["packet_length_total"]+pcap_file[:-8]+'_packet_length.csv'
    d=config.HTTPS_CONFIG["time_interval_total"]+pcap_file[:-8]+'_time_interval.csv'
    e=config.HTTPS_CONFIG["payload_total"]+pcap_file[:-8]+'_payload.csv'

<<<<<<< HEAD
    with open(config.HTTPS_CONFIG["total_path"]+pcap_file[8:-16]+'.csv','a')as f:
        # f.write('id,'+','.join(attrs)+'\n')
        for (flow,i) in zip(flows.values(),range(len(flows))):
            # 只有长度大于20的流才会保留
            if flow.pkt_count>=20:
                tmp=("%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
                     %(proto_name(flow.sport,flow.dport),
                       flow.src,
                       flow.sport,
                       flow.dst,
                       flow.dport,
                       flow.proto,
                       str(set(flow.extension_servername_indication)).strip("set([])"),
                       flow.push_flag_ratio(),
                       flow.avrg_len(),
                       flow.avrg_payload_len(),
                       flow.pkt_count,
                       flow.avrg_inter_arrival_time(),
                       flow.kolmogorov(),
                       flow.shannon(),
                       flow.max_len(),
                       flow.min_len(),
                       flow.std_len(),
                       #len(flow.extension_signature_algorithms),
                       len(flow.cipher_suites),
                       flow.avrg_window(),
                       flow.max_window(),
                       flow.min_window(),
                       flow.var_window(),
                       max(flow.session_id_length),
                       flow.avrg_ip_ttl(),
                       flow.max_ip_ttl(),
                       flow.min_ip_ttl()
                       ))
                f.write(pcap_file[8:-5]+"_"+str(i)+","+tmp+"\n")
                print ("packet number:%d"%i)
        print ("finish pcap_file[8:-5]")

    with open(config.HTTPS_CONFIG["record_type_total"]+pcap_file[8:-16]+'_record_type.csv','a')as f:
        for (flow,i) in zip(flows.values(),range(len(flows))):
            # 只有长度大于20的流才会保留
            if flow.pkt_count>=20:
                tmp=padArray(flow.record_type,256)
                tmp=str(tmp).strip('[]')
                f.write(pcap_file[8:-5]+"_"+str(i)+","+tmp+"\n")
    with open(config.HTTPS_CONFIG["packet_length_total"]+pcap_file[8:-16]+'_packet_length.csv','a')as f:
        for (flow,i) in zip(flows.values(),range(len(flows))):
            # 只有长度大于20的流才会保留
            if flow.pkt_count>=20:
                tmp=padArray(flow.length,0)
                tmp=str(tmp).strip('[]')
                f.write(pcap_file[9:-5]+"_"+str(i)+","+tmp+"\n")

if __name__ == '__main__':
    pcap_files=os.listdir(config.HTTPS_CONFIG["pcap_path"])
    record_type_names=["id"]+["r_"+str(i) for i in range(128)]+['label']
    packet_length_names=["id"]+["c_"+str(i) for i in range(128)]+['label']
    softwares=set([pcap_file[8:-16] for pcap_file in pcap_files])
=======
    with open(a,'a')as f1,open(b,'a') as f2,open(c,'a')as f3,open(d,'a')as f4,open(e,'a')as f5:
        # f.write('id,'+','.join(attrs)+'\n')
        for (flow,i) in zip(flows.values(),range(len(flows))):
            # 只有长度大于20的流才会保留
            if flow.pkt_count>=20:
                tmp1=("%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
                     %(proto_name(flow.sport,flow.dport),
                       flow.src,
                       flow.sport,
                       flow.dst,
                       flow.dport,
                       flow.proto,
                       str(set(flow.extension_servername_indication)).strip("set([])").replace(",",""),
                       flow.push_flag_ratio(),
                       flow.avrg_len(),
                       flow.avrg_payload_len(),
                       flow.pkt_count,
                       flow.avrg_inter_arrival_time(),
                       flow.kolmogorov(),
                       flow.shannon(),
                       flow.max_len(),
                       flow.min_len(),
                       flow.std_len(),
                       #len(flow.extension_signature_algorithms),
                       len(flow.cipher_suites),
                       flow.avrg_window(),
                       flow.max_window(),
                       flow.min_window(),
                       flow.var_window(),
                       max(flow.session_id_length),
                       flow.avrg_ip_ttl(),
                       flow.max_ip_ttl(),
                       flow.min_ip_ttl()
                       ))
                f1.write(pcap_file[:-5]+"_"+str(i)+","+tmp1+"\n")

                tmp2=padArray(flow.record_type,64,256)
                tmp2=str(tmp2).strip('[]')
                f2.write(pcap_file[:-5]+"_"+str(i)+","+tmp2+"\n")

                tmp3=padArray(flow.length,64,0)
                tmp3=str(tmp3).strip('[]')
                f3.write(pcap_file[:-5]+"_"+str(i)+","+tmp3+"\n")

                tmp4=padArray(flow.inter_arrival_times,64,0)
                tmp4=str(tmp4).strip('[]')
                f4.write(pcap_file[:-5]+"_"+str(i)+","+tmp4+"\n")

                tmp5=padArray(flow.payload,4096,128)
                tmp5=str(tmp5).strip('[]')
                f5.write(pcap_file[:-5]+"_"+str(i)+","+tmp5+"\n")
                print ("packet number:%d"%i)
        print ("finish %s"%pcap_file)

def run(pcaps):
    for p in pcaps:
        parse(p)

if __name__ == '__main__':
    pcap_files=os.listdir(config.HTTPS_CONFIG["pcap_path"])

    record_type_names=["id"]+["r_"+str(i) for i in range(128)]+['label']
    packet_length_names=["id"]+["c_"+str(i) for i in range(128)]+['label']
    time_interval_names=["id"]+["t_"+str(i) for i in range(128)]+['label']
    payload_names=["id"]+["p_"+str(i) for i in range(4096)]+['label']
    softwares=set([pcap_file[:-8] for pcap_file in pcap_files])

>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d
    for software in softwares:
        with open(config.HTTPS_CONFIG["total_path"]+software+'.csv','w+')as f:
            f.write('id,'+','.join(attrs)+'\n')
        with open(config.HTTPS_CONFIG["record_type_total"]+software+'_record_type.csv','w+')as f:
            f.write(','.join(record_type_names)+'\n')
        with open(config.HTTPS_CONFIG["packet_length_total"]+software+'_packet_length.csv','w+')as f:
            f.write(','.join(packet_length_names)+'\n')
<<<<<<< HEAD
=======
        with open(config.HTTPS_CONFIG["time_interval_total"]+software+'_time_interval.csv','w+')as f:
            f.write(','.join(time_interval_names)+'\n')
        with open(config.HTTPS_CONFIG["payload_total"]+software+'_payload.csv','w+')as f:
            f.write(','.join(payload_names)+'\n')
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d

    threads=[]

    for software in softwares:
        all_pcap=[i for i in pcap_files if i.find(software)>-1]
        t=threading.Thread(target=run,args=(all_pcap,))
        print ("start:%s"%software)
        t.start()
        threads.append(t)
    for k in threads:
        k.join()
