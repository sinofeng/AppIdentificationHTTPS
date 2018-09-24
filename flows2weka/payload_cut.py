#/usr/bin/env python
#coding=utf-8
import os
names=os.listdir("./payload")
for name in names:
    lines=""
    with open("./payload/"+name,'r')as f:
        lines=f.readlines()
    with open("./payload2/"+name,'w+')as ff:
        for line in lines:
            ff.write(line[6:38]+"\n")