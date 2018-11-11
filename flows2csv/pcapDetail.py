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

a=rdpcap("test.pcap")
a[1].pdfdump()
a[1].psdump("test.eps")