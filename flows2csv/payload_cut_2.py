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
            if nums<256:
                tmp=line[6:38]
                outcome+=tmp
                nums+=1
        newline=[]
        index=[i for i in range(8192) if i%2==0]
        for i in index:
            m=outcome[i]
            n=outcome[i+1]
            tmp1=(ord(m)-55) if (m in "ABCDEF") else (ord(m)-48)
            tmp2=(ord(n)-55) if (n in "ABCDEF") else (ord(n)-48)
            newline.append(tmp1*16+tmp2)
        ff.write(str(newline).strip('[]'))