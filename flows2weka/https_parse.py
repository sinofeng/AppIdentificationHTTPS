#!/usr/bin/env python
#coding=utf-8
import sys
import hashlib
from hashlib import md5
from scapy.all import *
from scapy.all import IP,TCP
from tcp_stream import TCPStream
import argparse

# 五元组确定一条流
def create_forward_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.src,pkt.sport,pkt.dst,pkt.dport,pkt.proto)
def create_reverse_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.dst,pkt.dport,pkt.src,pkt.sport,pkt.proto)
def create_flow_keys(pkt):
    return create_forward_flow_key(pkt),create_reverse_flow_key(pkt)

flows = {}
attrs = ['src','sport','dst','dport','proto','push_flag_ratio','average_len','average_payload_len','pkt_count','flow_average_inter_arrival_time','kolmogorov','shannon']

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

def main(pcap_file,output_type):
    packets=rdpcap("./raw_pcap/"+pcap_file)
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

    if output_type == "arff":
        print ("@relation protocol_detection")
        print ("@attribute protocol-name","{ssh,http,mysql,unknown}")

        for attr in attrs:
            if attr in ['pkt_count','average_len','flow_average_inter_arrival_time','push_flag_ratio','average_payload_len','kolmogorov','shannon']:
                print ("@attribute",attr,"numeric")
            else:
                print ("@attribute",attr,"string")
        print ("@data")
    else:
        attrs.insert(0,"protocol_name")
        print (','.join(attrs))

    with open('./output/'+pcap_file+'.csv','w+')as f:
        f.write('id,'+','.join(attrs)+'\n')
        for (flow,i) in zip(flows.values(),range(len(flows))):
            tmp=("%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s,%s,%s"%(proto_name(flow.sport,flow.dport),flow.src,flow.sport,flow.dst,flow.dport,flow.proto,flow.push_flag_ratio(),flow.avrg_len(),flow.avrg_payload_len(),flow.pkt_count,flow.avrg_inter_arrival_time(),flow.kolmogorov(),flow.shannon()))
            with open("./payload/"+pcap_file+"_"+str(i)+'.txt','w+')as ff:
                ff.write(hexdump(flow.payload,dump=True))
            f.write(str(i)+","+tmp+"\n")

    names=os.listdir("./payload")
    for name in names:
        lines=""
        with open("./payload/"+name,'r')as f:
            lines=f.readlines()
        with open("./payload2/"+name,'w+')as ff:
            for line in lines:
                ff.write(line[6:38]+"\n")

if __name__ == '__main__':
    pcap_files=os.listdir('./raw_pcap')
    for pcap_file in pcap_files:
        main(pcap_file,'csv')