#/usr/bin/env python
#coding=utf-8
import os

#每一个文件对应的是一个flow的全部负载信息
names=os.listdir("./payload")
for name in names:
    lines=""
    with open("./payload/"+name,'r')as f:
        lines=f.readlines()
    nums=0
    with open("./payload2/"+name,'w+')as ff:
        outcome=""
        for line in lines:
            #只保存前128行数据,将128行数据拼接为一行
            if nums<128:
                tmp=line[6:38]
                outcome+=tmp
                nums+=1
        outcome='{:G<4096}'.format(outcome[0:4096])
        newline=[ord(s)-55 if s in "ABCDEFG" else ord(s)-48 for s in outcome]
        ff.write(str(newline).strip('[]'))


# 将产生的负载写入同一个文件，代码有待优化
# python文件拼接
# Android_aiqiyi_2018091901  0-531
# Android_cloudmusic_2018091901  0-652
# Android_shoujibaidu_2018091901  0-364
with open("./payload3/Android_shoujibaidu_20180919011.csv",'w+') as f:
    names=['Android_shoujibaidu_20180919011.pcap_%s.txt'%str(i) for i in range(365)]
    lines=""
    for name in names:
        with open('./payload2/'+name,'r')as ff:
            lines+=(ff.readline()+'\n')
    f.write(lines)