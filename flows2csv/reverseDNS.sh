#!/usr/bin/env bash
echo "############# Reverse DNS ##############"
while read id servername;do
    ip=$(nslookup $servername | grep ^Name -A1 | grep Address | awk '{printf ($2" ")}');
    echo "$id,$servername,$ip";
done<$1 >$2

