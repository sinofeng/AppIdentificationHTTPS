import re
import xml.etree.ElementTree as ET
import os

def get_tcp_stream(name):
    tree = ET.parse("./raw_xml/"+name)
    root = tree.getroot()
    packets = root.getchildren()
    streams = {}
    for packet in packets:
        try:
            streams_num = packet[4][4].get("showname").split(" ")[-1]#tcp is the fourth 
            if streams.get(streams_num)==None:
                streams[streams_num]=[packet]
            else:
                streams[streams_num].append(packet)
        except Exception:
            continue
    finnal_datas = []
    for key in streams.keys():
        ip_info = streams[key][0][3].get("showname").split(" ")  #get ip and port
        ip_src_host = ip_info[5]
        ip_dst_host = ip_info[8]
        port_info = streams[key][0][4].get("showname").split(" ")
        tcp_src_port = port_info[6][1:-2]
        tcp_dst_port = port_info[10][1:-2]
        ip_version = streams[key][0][3][0].get("show")
        print (ip_src_host,ip_dst_host,tcp_src_port,tcp_dst_port,ip_version)
        ip_and_post_info = [ip_src_host,ip_dst_host,tcp_src_port,tcp_dst_port,ip_version]
        # 负载
        application_datas = []
        client_hello_features=[]
        server_hello_features=[]
        app_data=""
        for packet in streams[key]:
            try:    
                if "Client Hello" in packet[5][0].get("showname"):
                    print (packet[5][0][3][7].get("showname"),packet[5][0][3][9].get("showname"))
                    client_hello_features = [packet[5][0][3][7].get("showname"),packet[5][0][3][9].get("showname")]
                    continue
                if "Server Hello" in packet[5][0].get("showname"):
                    print (packet[5][0][3][6].get("showname"),packet[5][0][3][7].get("showname"))
                    server_hello_features = [packet[5][0][3][6].get("showname"),packet[5][0][3][7].get("showname")]
                    continue
                print (packet[0][-1].get("value"),packet[0][1].get("size"),packet[5][0][2].get("show"))
                if packet[5][-1][-1].get("name")!="ssl.app_data":
                    continue
                app_data = packet[5][-1][-1].get("value")
            
                # 将application data补齐
                if len(app_data)>=1024:
                    app_data = app_data[0:1024]
                else:
                    app_data = app_data+"g"*(1024-len(app_data))

                application_datas.append([packet[0][-1].get("value"),packet[0][1].get("size"),packet[5][0][2].get("show"),app_data])
        
            except Exception:
                continue
        finnal_datas.append([ip_and_post_info,client_hello_features,server_hello_features,application_datas])

    if not os.path.exists("./processed_data/"+name[:-14]):
        os.makedirs("./processed_data/"+name[:-14])
    i=0
    # 每一条流存为一个文件
    for datas in  finnal_datas:
        with open("./processed_data/"+name[:-14]+"/"+name[:-4]+str(i)+".txt","w") as f:
            f.write(" ".join(datas[0])+"\n")
            f.write(" ".join(datas[1])+"\n")
            f.write(" ".join(datas[2])+"\n")
            for application_data in datas[3]:
                f.write(" ".join(application_data)+"\n")
    i+=1

if __name__ == '__main__':
	os.chdir('I:/Research/paper/data')
	data_dir="./raw_xml/"
	names=os.listdir(data_dir)
	for name in names:
		get_tcp_stream(name)
        
