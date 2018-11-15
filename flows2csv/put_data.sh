#!/usr/bin/env bash
echo "start upload..."
sftp tm@tm<<EOF
put /home/ss/workplace/experiment/data/wd_https/https_train_eval/packet_payload* /media/data/tm/

quit
EOF
echo "success!"
