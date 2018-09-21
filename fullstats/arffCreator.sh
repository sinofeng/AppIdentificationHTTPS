#!/bin/zsh

PREFIX="$1"
if [[ -f "$PREFIX"filelist ]]
then
  :
else
  echo "Check prefix \"$PREFIX\"" 1>&2
  exit 2
fi

export TMPDIR="${PREFIX}tmp-$PPID-$$-${RANDOM}${RANDOM}${RANDOM}.d"
mkdir "$TMPDIR"
export TMP="$TMPDIR"
export TMPPREFIX="$TMPDIR/zsh-$PPID-$$-${RANDOM}${RANDOM}${RANDOM}-"

perl -lpe 'BEGIN{open SENSES, "senses" or die "open senses"; chomp(@_=<SENSES>); for(@_){ m<^(6/\d+\.\d+\.\d+\.\d+/\S+) (\S+)$> or die "pattern"; if ($2 eq "client/server") { $h{"out/$1"}=1 } elsif ($2 eq "server/client") {$h{"out/$1"}=0} } close SENSES or die "close" } chomp($o=$_); s<^output/\d+/([0-9\.]+)/([0-9\.]+)/([0-9\.]+)-(\d+)-(\d+).trace$><$1 $4 $2 $5 $3> && s< (\d{8,})\.(\d+)><" ".$1.".".(0 x (6-length($2))).$2>e or
           s<^out/\d+/(\d+\.\d+\.\d+\.\d+)/(\d{5})/(\d+\.\d+\.\d+\.\d+)/(\d{5})/(\d{7,}\.[0-9]{6})$><$1 $2 $3 $4 $5> && ++$new && do { if (exists $h{$o}) { s/^(\S+\s+\S+)\s+(\S+\s+\S+)\s+/$2 $1 / if $h{$o} == 1; } else {die "hash did not exist"}; 1; } or
            die "pattern match fails at line $. for <$_>"' < "${PREFIX}"filelist > "${PREFIX}"basic.subset || exit 1 # old format

perl -alne 'BEGIN{ sub invalid { (scalar grep {$_ ne "NA" and $_ ne "N" and $_ ne "Y"} (@F[34..37, 40, 41]) or scalar grep {not m/^\d+\.\d{6}$/ and $_ ne "NA"} @F[76..79,84..93,96..103,114..121] or scalar grep {not m/^\d+$/ and $_ ne "NA"} @F[0..33,38,39,42..75,80..83,94..95,104..113] or scalar grep {$_ ne "N" and $_ ne "NA" and $_ ne "Y"} @F[34..37,40,41] ) } }   if(@F==120){if ($F[34] ne "NA" or $F[35] ne "NA"){die}; $F[34]=$F[35]="NA NA"} elsif (@F==124) { @F[2..3]=() } elsif (@F == 0) { @F=("NA")x122 } elsif (@F != 122) { die } else {if (&invalid()) {  if ($F[36] ne "NA" or $F[37] ne "NA"){die}; $F[34] = $F[35] ="NA NA"; @F[2..3]=(); $_=join " ",@F; split; @F=@_; if(&invalid()){die}}} ; $_=join " ", @F; split; @F = @_; if (@F != 122) { die "pattern <$_> line $.: number of fields is ".(scalar @F); @F=("NA")x122}; for(34,35,36,37,40,41) { if($F[$_] =~ m/\d/) { die "digit exists in wrong place at field$_ for pattern <@F> at line $." } } ; print join " ",@F' < "${PREFIX}"tcptrace.char > "${PREFIX}"tcptrace.char.mod

paste -d' ' \
  =(cut -d' ' -f4 "${PREFIX}"transfer.mode) \
  =(cut -d' ' -f1,2 "${PREFIX}"basic.subset) \
  =(cut -d' ' -f2,19 "${PREFIX}"tcptrace.char.mod) \
  =(cut -d' ' -f3,4 "${PREFIX}"basic.subset) \
  =(cut -d' ' -f1,18 "${PREFIX}"tcptrace.char.mod) \
  > "${PREFIX}"basic.char

if [[ -f "complex_category2" ]]
then
  perl -ane '$h{$F[9]}=$F[0]; END{(open _F, "'"${PREFIX}"filelist'") && (open _SSC, ">'"${PREFIX}"subsubclasses'") &&  (open _SC, ">'"${PREFIX}"subclasses'") && (open _C, ">'"${PREFIX}"classes'") or die "an open fails";  while(<_F>){ chomp; s<.*/(\d{8,}\.\d+).*><$1>; die "no match for <$_> at line $." unless exists $h{$_}; $_=$h{$_}."\n"; print _SSC; s/(ATTACK|GAMES|MULTIMEDIA|P2P)/OTHER/; print _SC; s/FTP-(PASV|DATA|CONTROL)/BULK/; print _C};close _F && close _SSC && close _SC && close _C or die "close fails"}' < complex_category2 || exit 1
else
  if [[ -f "myclasses" ]]
  then
    cat "myclasses" | tee "${PREFIX}"subsubclasses "${PREFIX}"subclasses > "${PREFIX}"classes
  else
    yes NIL | head -$(echo $(wc -l < "$PREFIX"filelist)) | tee "${PREFIX}"subsubclasses | tee "${PREFIX}"subclasses > "${PREFIX}"classes
  fi
fi

paste -d' ' \
 =(cut -d' ' -f5 "${PREFIX}"basic.subset) \
 "${PREFIX}"basic.char \
 "${PREFIX}"classes \
 "${PREFIX}"time.stats.all \
 "${PREFIX}"data.stats.all \
 =(cut -d' ' -f5 "${PREFIX}"basic.subset) \
 "${PREFIX}"tcptrace.char.mod \
 "${PREFIX}"data.stats.a_b \
 "${PREFIX}"data.stats.b_a \
 "${PREFIX}"time.stats.a_b \
 "${PREFIX}"time.stats.b_a \
 "${PREFIX}"last_connection \
 "${PREFIX}"subclasses \
 "${PREFIX}"transfer.mode \
 "${PREFIX}"eff.band.all \
 "${PREFIX}"eff.band.a_b \
 "${PREFIX}"eff.band.b_a \
 "${PREFIX}"fft.all \
 "${PREFIX}"fft.a_b \
 "${PREFIX}"fft.b_a \
 "${PREFIX}"subsubclasses \
  | perl -ne 'chomp; s/NA/?/g; s/\s+/ /g; print "$_\n"; split; die "FAILED NUMBER OF COLUMNS <$_> at line $." unless @_ == 266' \
  > "${PREFIX}"filelist.all || exit 1

tr ' ' , < "${PREFIX}"filelist.all \
 | perl -e'print "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266\n";foreach $line (<STDIN>){print $line}' \
 > "${PREFIX}"filelist.all.csv

#---------------------------------
#keeps the features as identified by Moore(2004) Full Features set.  

cut -d',' -f4,8,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,210,211,212,213,214,215,216,218,219,220,221,222,223,224,225,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266 < "${PREFIX}"filelist.all.csv > "${PREFIX}"filelist.weka.allclass.csv

#Patch Fedora Core 5 tail 5.6 instead of tail +2 -> tail -n +2

#Create the arff file for the full feature set
(cat my.real.weka.header; tail -n +2 "${PREFIX}"filelist.weka.allclass.csv) > "$PREFIX"filelist.weka.allclass.arff # for weka toolkit

#---------------------------------
#keeps the features as identified by Moore(2004) awmreduced set

#From Fix.java = 59,94,95,84,161,45,165,83,113,58,248 
cut -d',' -f60,95,96,85,162,46,166,84,114,59,249 < "${PREFIX}"filelist.weka.allclass.csv > "${PREFIX}"filelist.weka.awmreduced.csv



#Create the awmreduced feature set arff file
(cat my.real.weka.header.reduced; tail -n +2 "${PREFIX}"filelist.weka.awmreduced.csv) > "$PREFIX"filelist.weka.awmreduced.arff


rmdir "$TMPDIR" || exit 1
