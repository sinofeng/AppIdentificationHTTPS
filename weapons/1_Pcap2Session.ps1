foreach($f in gci 1_Pcap *.pcap)
{
    SplitCap -p 100000 -b 100000 -r $f.FullName -o 2_Session\AllLayers\$($f.BaseName)-ALL
    SplitCap -p 100000 -b 100000 -r $f.FullName -s flow -o 2_Session\AllLayers\$($f.BaseName)-ALL
    gci 2_Session\AllLayers\$($f.BaseName)-ALL | ?{$_.Length -eq 0} | del

    SplitCap -p 100000 -b 100000 -r $f.FullName -o 2_Session\L7\$($f.BaseName)-L7 -y L7
    SplitCap -p 100000 -b 100000 -r $f.FullName -s flow -o 2_Session\L7\$($f.BaseName)-L7 -y L7
    gci 2_Session\L7\$($f.BaseName)-L7 | ?{$_.Length -eq 0} | del
}

finddupe -del 2_Session\AllLayers
finddupe -del 2_Session\L7