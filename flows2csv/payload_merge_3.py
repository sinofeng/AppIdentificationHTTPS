#coding=utf-8
# 将产生的负载写入同一个文件，代码有待优化
import os
payload2="./payload2/"
filenames=os.listdir(payload2)
softwares=["aiqiyi","cloudmusic","shoujibaidu","guotaijunan","weibo","xiecheng"]
filenames=[[tmp for tmp in filenames if tmp.find(software)>-1] for software in softwares]
print(filenames)

payload3="./payload3/"

for software_files in filenames:
    tmp=""
    for software_file in software_files:
        with open(payload2+software_file)as ff:
            tmp+=str(ff.readline()).strip("['']")+"\n"

    index=software_files[0].find("_2018")
    with open(payload3+software_files[0][0:index],'w+')as f:
        f.write(tmp)