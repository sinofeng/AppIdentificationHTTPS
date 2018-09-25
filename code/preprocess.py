#!/usr/bin/env python
# coding=utf-8
import os
import re

# xml--->txt:抽取加密负载数据
xml_filenames=os.listdir("./raw_xml/")
for xml_filename in xml_filenames:
    context=''
    with open("./raw_xml/"+xml_filename,'r') as f:
        context=f.read()
    m = re.findall('Encrypted Application Data:.*?value="(.*?)"',context)
    with open("./raw_txt/"+xml_filename[:-4]+".txt",'a+')as new_txt_file:
        for i in m:
            new_txt_file.write(i+'\n')

filenames=os.listdir("./raw_txt")

filename_label={filenames[i]:i for i in range(len(filenames))}
print("Application label:"+str(filename_label))
# 将抽取出来的加密数据进行截取/填充，保存为1024长度，空位使用g(16)补齐
# 将字符转换为浮点数，即没一个样本是一个1024长度的向量
for filename in filenames:
    with open("./processed_data/"+filename+".csv","a+") as newfile:
        with open("./raw_txt/"+filename,'r',encoding='utf-8')as f:
            lines=f.readlines()
            for line in lines:
                ss='{:g<1024}'.format(line.strip('\n')[0:1024])
                # 字符转换为数字
                newline=[ord(s)-55 if s in "ABCDEF" else ord(s)-48 for s in ss]
                # 将应用标签表示为数字标签
                newfile.write(str(newline).strip('[]')+','+str(filename_label[filename])+"\n")



