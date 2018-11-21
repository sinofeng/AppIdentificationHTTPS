#/usr/bin/env python
#coding=utf-8

"""
问题：
(0) 在测试中，3123个packet中有590个packet包含handshake字段
(1) 握手的过程是不是在同一条流中？是在一个流中，反证：不在一个流中意味着五元组不会相同，无法完成握手。
(2) 一个pcap文件中有多少次的握手？ok!
(3) 如何获取一个完整的握手过程？
(4) 握手过程不同如何处理？
(5) 同一个流中可能有多次的握手，这里可以得到同一个流对应的packet，则很容易得到有多少个握手（如，可以统计client hello的个数）。
(6) 单独看流中握手的个数可能并不太科学，因为流的大小不一，另外还有丢包存在。
(7) 对于当前给定的packet需要判断其属于哪个阶段的packet，即如何区分packet的类别？(目前想法是直接判断其包含的关键字)
"""

from scapy_ssl_tls.ssl_tls import *
p=rdpcap("./raw_pcap/test.pcap")
print(p[0].show())
print("--------------------------------------------------------------------------------------")
print(p[1].show())
print("--------------------------------------------------------------------------------------")
print(p[2].show())
print("--------------------------------------------------------------------------------------")
print(p[3]["SSL/TLS"].show())
print("--------------------------------------------------------------------------------------")
pkt=p[3]["SSL/TLS"]["TLS Handshakes"]["TLS Client Hello"]
print(pkt.show())
print("----------------------------------")
print(pkt.version)
print("----------------------------------")
print(str(pkt.cipher_suites_length))
# 以上可以将各个HTTPS字段提取出来
print("--------------------------------------------------------------------------------------")

output="./output/"
# for i in range(10):
#     with open(output+str(i)+".txt",'w+') as f:
#         f.write(str(p[i].sprintf)
handshake_pkt=[]
for i in range(3123):
    if((str(p[i].sprintf).find("Handshake")>-1)):
        handshake_pkt.append(i)
print("handshake_pkt:")
print(handshake_pkt)
print("handshake_pkt nums:%d"%len(handshake_pkt))
# 抽取流
# 五元组确定一条流,流中的packet过少的话应该被当作脏数据剔除，因为我们在抓包的时候必然损失了一些数据，并不能全部解析完整的过程
def create_forward_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.src,pkt.sport,pkt.dst,pkt.dport,pkt.proto)
def create_reverse_flow_key(pkt):
    return "%s:%s->%s:%s:%s"%(pkt.dst,pkt.dport,pkt.src,pkt.sport,pkt.proto)
def create_flow_keys(pkt):
    return create_forward_flow_key(pkt),create_reverse_flow_key(pkt)
flows={}
# 以下将同一个流的packet编号写在一起(这里处理的是双向的流，单向只要指定一个端口是443)
for i in range(len(p)):
    key,reverse_key = create_flow_keys(p[i]["IP"])
    if key in flows.keys():
        flows[key].append(i)
    elif reverse_key in flows.keys():
        flows[reverse_key].append(i)
    else:
        flows[key]=[i]
pkt_num=0
flow_length=[]
for key in flows.keys():
    num=len(flows[key])
    flow_length.append(num)
    pkt_num+=num
print("pkt_num:%d"%pkt_num)
print("flow_length:")
print(flow_length)
print("how many flows:%d"%len(flow_length))
print ("flows:")
print(flows)

"""
基于流构建特征库：
1.一个流+软件的标签，作为一个样本。即样本以流为单位。
2.分阶段，将数据格式进行统一，如Handshake 可以分为client hello,server hello等。
3.每个流中的packet有相应的时序特征，需要提取和保留。
4.特征包括数值和离散值，注意处理。
5.人为构造特征，如packet长度均值，信息熵等。参考248种特征。
"""