#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: pcapDetail.py
@time: 18-11-10 下午4:14
@desc:
"""

from scapy_ssl_tls.ssl_tls import *
from struct import unpack
a=rdpcap("test.pcap")
# a[1].pdfdump()
# a[1].psdump("test.eps")
print(a[0].payload)
print("*"*50)
print(a[0]["TCP"].payload)
tmp=hexdump(a[0].payload)
print("*"*50)
hexdump(a[0]["TCP"].payload)
print("*"*50)
s=str(a[0]["TCP"].payload)
print(s)
print("*"*50)
C=[ord(c) for c in s]
print(C)
print(len(C))