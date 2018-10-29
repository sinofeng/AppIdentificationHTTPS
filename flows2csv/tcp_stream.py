#coding=utf-8
from scapy.all import *
import scapy
from numpy import *
from entropy import kolmogorov, shannon
import numpy as np
#We are assuming:
#1) Its an IP packet
#2) Its an TCP packet
def sni(pkt):
	if str(pkt.sprintf).find("server_names")>-1:
		return str(pkt["TLS Servername"].data)
# def algs(pkt):
# 	if str(pkt.sprintf).find("algs")>-1:
# 		return pkt["TLS Extension Signature Algorithms"].algs
def ciphers(pkt):
	if str(pkt.sprintf).find("TLSClientHello")>-1:
		return pkt["TLS Client Hello"].cipher_suites
def cipher(pkt):
	if str(pkt.sprintf).find("TLSServerHello")>-1:
		try:
			return pkt["TLS Server Hello"].cipher_suite
		except:
			pass
def window(pkt):
	return pkt["TCP"].window
def session_id_len(pkt):
	fields=["TLS Client Hello","TLS Server Hello"]
	for field in fields:
		try:
			return pkt[field].session_id_length
		except:
			pass
	return None
def client_extensions_length(pkt):
	if str(pkt.sprintf).find("TLSClientHello")>-1:
		try:
			return pkt["TLS Client Hello"].extensions_length
		except:
			pass
def server_extensions_length(pkt):
	if str(pkt.sprintf).find("TLSServerHello")>-1:
		try:
			return pkt["TLS Server Hello"].extensions_length
		except:
			pass
def session_ticket_lifetime(pkt):
	if str(pkt.sprintf).find("TLSSessionTicket")>-1:
		return pkt["TLS Session Ticket"].lifetime

def mode(nums):
	counts=np.bincount(nums)
	return np.argmax(counts)
def statistics(nums):
	return np.mean(nums),np.max(nums),np.min(nums),np.median(nums),np.var(nums)

def getRecordType(pkt):
	try:
		recordtype=pkt["TLS Record"].content_type
	except:
		try:
			recordtype=pkt["SSLv2 Record"].content_type
		except:
			recordtype=256
	if 1==0:
		return 99
	else:
		return recordtype

class TCPStream:
	def __init__(self,pkt):
		self.src = pkt.src 
		self.dst = pkt.dst
		self.flags = [pkt.sprintf("%TCP.flags%")]
		self.sport = pkt.sport
		self.dport = pkt.dport        
		self.time = pkt.time
		self.proto = pkt.proto
		self.inter_arrival_times = [0]
		self.pkt_count = 1
		self.len = pkt.len # packet的累加
		self.length=[pkt.len] # packet单个长度
		self.payload = str(pkt["TCP"].payload)
		self.pkt = pkt
		self.extension_servername_indication=[sni(pkt)]
		# self.extension_signature_algorithms=[algs(pkt)]
		self.cipher_suites=[ciphers(pkt)]
		self.cipher_suite=[cipher(pkt)]
		self.tcp_window=[window(pkt)]
		self.session_id_length=[session_id_len(pkt)]
		self.ip_ttl=[pkt["IP"].ttl]
		self.tls_session_ticket_lifetime=[session_ticket_lifetime(pkt)]
		self.client_hello_extensions_length=[client_extensions_length(pkt)]
		self.server_hello_extensions_length=[server_extensions_length(pkt)]
		self.ip_chksum=[pkt["IP"].chksum]
		self.tcp_options=[pkt["TCP"].options] # 是否存在以及options的数量，可能是强特！
		self.record_type=[getRecordType(pkt)]


	def unique_flags(self):
	    seen = set()
	    for item in self.flags:
	        if item not in seen:
	            seen.add( item )
		    yield item

	def avrg_len(self):
		return self.len/self.pkt_count
	def max_len(self):
		return max(self.length)
	def min_len(self):
		return min(self.length)
	def std_len(self):
		return np.var(self.length)
	def kolmogorov(self):
		return round(kolmogorov(self.payload),4)
	def shannon(self):
		return round(shannon(self.payload),4)
	def avrg_payload_len(self):
		return len(self.payload)/self.pkt_count
	def avrg_inter_arrival_time(self):
		return round(mean(self.inter_arrival_times),4)
	def min_inter_arrival_time(self):
		return min(self.inter_arrival_times)
	def max_inter_arrival_time(self):
		return max(self.inter_arrival_times)
	def var_inter_arrival_time(self):
		return np.var(self.inter_arrival_times)
	def median_inter_arrival_time(self):
		return np.median(self.inter_arrival_times)
	def avrg_window(self):
		return mean(self.tcp_window)
	def max_window(self):
		return max(self.tcp_window)
	def min_window(self):
		return min(self.tcp_window)
	def var_window(self):
		return np.var(self.tcp_window)
	def avrg_ip_ttl(self):
		return mean(self.ip_ttl)
	def max_ip_ttl(self):
		return max(self.ip_ttl)
	def min_ip_ttl(self):
		return min(self.ip_ttl)
	def var_ip_ttl(self):
		return np.var(self.ip_ttl)
	def push_flag_ratio(self):
		return len([ f for f in self.flags if 'P' in f ]) / float(len(self.flags))


	def add(self,pkt):
		self.pkt_count += 1
		self.len += pkt.len
		self.length.append(pkt.len)
		self.inter_arrival_times.append(pkt.time - self.time)
		self.flags.append(pkt.sprintf("%TCP.flags%"))
		self.payload += str(pkt["TCP"].payload)
		self.pkt = pkt
		sni_tmp=sni(pkt)
		if sni_tmp!=None:
			self.extension_servername_indication.append(sni_tmp)
		# algs_tmp=algs(pkt)
		# if algs_tmp!=None:
		# 	self.extension_signature_algorithms.append(algs_tmp)
		ciphers_tmp=ciphers(pkt)
		if ciphers_tmp!=None:
			self.cipher_suites.append(ciphers_tmp)
		cipher_tmp=cipher(pkt)
		if cipher_tmp!=None:
			self.cipher_suite.append(cipher_tmp)
		self.tcp_window.append(window(pkt))
		session_len=session_id_len(pkt)
		if session_len!=None:
			self.session_id_length.append(session_len)
		self.ip_ttl.append(pkt["IP"].ttl)
		session_ticket_lifetime_tmp=session_ticket_lifetime(pkt)
		if session_ticket_lifetime_tmp!=None:
			self.tls_session_ticket_lifetime.append(session_ticket_lifetime_tmp)
		client_extensions_length_tmp=client_extensions_length(pkt)
		if client_extensions_length_tmp!=None:
			self.client_hello_extensions_length.append(client_extensions_length_tmp)
		server_extensions_length_tmp=server_extensions_length(pkt)
		if server_extensions_length_tmp!=None:
			self.server_hello_extensions_length.append(server_extensions_length_tmp)
		self.ip_chksum.append(pkt["IP"].chksum)
		self.tcp_options.append(pkt["TCP"].options)
		self.record_type.append(getRecordType(pkt))

	def remove(self,pkt):
		raise Exception('Not Implemented')
