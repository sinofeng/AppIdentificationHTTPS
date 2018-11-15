#!/usr/bin/env bash
echo "start download..."
sftp tianmao@S2<<EOF
get -r /home/tianmao/data/wd_https/output/total/ /home/ss/workplace/experiment/data/wd_https/output/
get -r /home/tianmao/data/wd_https/packet_length/total/ /home/ss/workplace/experiment/data/wd_https/packet_length/
get -r /home/tianmao/data/wd_https/payload/total/ /home/ss/workplace/experiment/data/wd_https/payload/
get -r /home/tianmao/data/wd_https/record_type/total/ /home/ss/workplace/experiment/data/wd_https/record_type/
get -r /home/tianmao/data/wd_https/time_interval/total/ /home/ss/workplace/experiment/data/wd_https/time_interval/
quit
EOF
echo "success!"

