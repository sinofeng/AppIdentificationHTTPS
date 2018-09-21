#!/usr/bin/perl 

#Full Stats Scripts 
#Author: Kaysar Abdin (NRL)

use strict;
use Getopt::Long;
use Statistics::Descriptive;
use Math::Trig;
use FileHandle;
use IPC::Open2;
use POSIX qw(floor);
$| = 1; STDERR->autoflush(1); STDOUT->autoflush(1);
use Shell qw(rm mkfifo);

# Packet quota using tcpdump
my $opt_pkt; # -p N (number of packet)
# Duration using tcpslice
my @opt_dur; # -d start finish (default 0 100)

my @tcptrace_data;
# tcpdump variables:
my $count = 0;
# tcpslice variables:
my $start = 0; # default 0, meaning start of flow
my $end; # duration

mkfifo("pipe tmpipe"); # pipes for paket and duration selection

my $path_tcpslice = "tcpslice"; # tcpslice for duration
my $path_tcptrace = "tcptrace"; # tcptrace for features
my $path_tcpdump = "tcpdump";   # tcpdump for packet count and features
my $path_mtkdemo = "Mtk_sa";    # Mtk_sa for computation of EFFBW
my $path_fft = "f"; # Fast Fourier Analysis - computation of FFT
my $err;
my $offset; my $offset_ab; my $offset_ba;
my $prefix = "output_".substr(time().$$.getppid().substr(rand(),2).substr(rand(),2),0,20);

my $opt_old; # specify ``--old'' to assume the old format of directory layout
my @output_lastconnection;

$_ = int GetOptions ("old" => \$opt_old,'p=i' => \$opt_pkt,'d=i'=>\@opt_dur) or die "GetOptions: $!($_)";

#$_=int GetOptions ("old"  => \$opt_old) or die "GetOptions: $!($_)";

chomp (my @files = <>); # list of files to be processed, given as a relative path

if ($opt_pkt){
        print "packet option\n";
        $count = $opt_pkt;
        print "$count\n"
}
if (@opt_dur){
        if (@opt_dur > 1){
                print "Duration option start and finish\n";
                print "more than one\n";
                $start = $opt_dur[0];
                $end = $opt_dur[1];
                print "$start $end\n";
        }
        else{
                print "duration option (default start 0)\n";
                $end = $opt_dur[0];
                print "$start $end\n";
        }
}

my %times_by_ip;
my @time_by_ln;

my ($ip, $src_ip, $dst_ip, $time); # note: the sense of src and dst may be reversed in old/new version (to, perhaps, client/server)

for my $line_number (0..$#files) {
  $_ = $files [$line_number];
  if ($opt_old) {
    #                             output/protocol/ipsvr/ipclt/utime-portclt-portip.trace
    ($src_ip, $dst_ip, $time) = m<^output/\d+/([0-9\.]+)/([0-9\.]+)/([0-9\.]+)-\d+-\d+.trace$>
      or die "file \"$_\" does not match pattern (old format)";
    $time =~ s<\.(\d+)> <".".(0 x (6-length($1))).$1>e; # fix timestamp (convert %d.%d to %d.%06d)
  } else {
    #                            out/protocol/ipsvr/portsvr/ipclt/portclt/utime 
    ($src_ip, $dst_ip, $time) = m<^out/\d+/([0-9\.]+)/\d{5}/([0-9\.]+)/\d{5}/([0-9\.]+)$>
      or die "file \"$_\" does not match pattern (new format)";
  } 

  $time_by_ln [$line_number] = $time;
  push @{$times_by_ip{ do { join " ", sort ($src_ip, $dst_ip) } }}, [ $time, $line_number ];
}

for $ip (keys %times_by_ip) {
  my $last;
  for (sort { $a->[0] <=> $b->[0] } @{$times_by_ip{$ip}}) {
    $output_lastconnection [$_->[1]] = $last ? (sprintf("%.06f", $_->[0] - $last)) : "?";
    $last = $_->[0];
  }
}

open OUT_LASTCONNECTION, ">${prefix}_last_connection" or die "open: $!";
for (0..$#files) { print OUT_LASTCONNECTION "$output_lastconnection[$_]\n" or die }
close OUT_LASTCONNECTION or warn;

open OUT_DATASTATSALL, ">${prefix}_data.stats.all" or die "open: $!";
open OUT_DATASTATSAB, ">${prefix}_data.stats.a_b" or die "open: $!";
open OUT_DATASTATSBA, ">${prefix}_data.stats.b_a" or die "open: $!";
open OUT_TIMESTATSALL, ">${prefix}_time.stats.all" or die "open: $!";
open OUT_TIMESTATSAB, ">${prefix}_time.stats.a_b" or die "open: $!";
open OUT_TIMESTATSBA, ">${prefix}_time.stats.b_a" or die "open: $!";
open OUT_TCPTRACECHAR, ">${prefix}_tcptrace.char" or die "open: $!";
open OUT_EFFBANDALL, ">${prefix}_eff.band.all" or die "open: $!";
open OUT_EFFBANDAB, ">${prefix}_eff.band.a_b" or die "open: $!";
open OUT_EFFBANDBA, ">${prefix}_eff.band.b_a" or die "open: $!";
open OUT_FFTALL, ">${prefix}_fft.all" or die "open: $!";
open OUT_FFTAB, ">${prefix}_fft.a_b" or die "open: $!";
open OUT_FFTBA, ">${prefix}_fft.b_a" or die "open: $!";
open OUT_TRANSFERMODE, ">${prefix}_transfer.mode" or die "open: $!";
open OUT_FILELIST, ">${prefix}_filelist" or die "open: $!";
open OUT_FILELISTPROCESSED, ">${prefix}_filelist.processed" or die "open: $!";
open OUT_FILELISTSKIPPED, ">${prefix}_filelist.skipped" or die "open: $!";
open OUT_FILELISTWARNINGS, ">${prefix}_filelist.warnings" or die "open: $!";
OUT_FILELISTPROCESSED->autoflush(1);
OUT_FILELISTSKIPPED->autoflush(1);
OUT_FILELISTWARNINGS->autoflush(1);

print OUT_FILELIST join "\n", (@files, '');
close OUT_FILELIST or warn;

my $total_size = 0;
my $start_time = time ();

for my $line_number (0..$#files) {
  my $size = (-s $files [$line_number]);
  if ($size <= 536870912) {
    $total_size += $size;
    my $brate = int ($total_size / (0.001 + time () - $start_time));
    my $frate = sprintf ("%.3f", ($line_number / (0.001 + time () - $start_time)));
    my $eta = int ($frate * ($#files - $line_number));
    $eta = sprintf ("%02d:%02d:%02d:%02d", floor($eta/86400), floor (($eta - 86400*floor($eta/86400))/3600), floor (($eta/60)%60), $eta % 60);
    print STDERR "\rfileno $line_number $size/$total_size brate $brate frate $frate \"$files[$line_number]\" eta $eta      \b\b\b\b\b\b";
    undef $offset; undef $offset_ab; undef $offset_ba;

    my @stat = map { Statistics::Descriptive::Full -> new () } qw( Wire IP Control );
    my @stat_ab = map { Statistics::Descriptive::Full -> new () } qw( Wire IP Control );
    my @stat_ba = map { Statistics::Descriptive::Full -> new () } qw( Wire IP Control );

    my (@all_data, @ab_data, @ba_data);
    my $total_frames = 0; my $total_frames_ab = 0; my $total_frames_ba = 0;

    if((@opt_dur)&&(!$opt_pkt)){ # duration option only
        my $pipe_tcpslice_pid = open PIPE_TCPSLICE, "$path_tcpslice $start $end  $files[$line_number] -w pipe |";
    }
    if(($opt_pkt)&&(!@opt_dur)){ # packet count option only
        my $pipe_tcpdump_pid = open PIPE_DUMP_TCPDUMP, "$path_tcpdump -c $count -r $files[$line_number] -w pipe 2>/dev/null |";
    }
    if((@opt_dur)&&($opt_pkt)){ # packet and duration option
        my $pipe_tcpslice_pid = open PIPE_TCPSLICE, "$path_tcpslice $start $end $files[$line_number] -w tmpipe |";
        my $pipe_tcpdump_pid = open PIPE_DUMP_TCPDUMP, "$path_tcpdump -c $count -r tmpipe -w pipe |";
    }
    if ((@opt_dur)||($opt_pkt)){ # options specified tcptrace
        my $pipe_tcptrace_pid = open PIPE_TCPTRACE, "$path_tcptrace -r -l -o1 -n -Q pipe |";
        chomp(@tcptrace_data = (<PIPE_TCPTRACE>));
    }
    if ((!@opt_dur)&&(!$opt_pkt)){ # no options specififed tcptrace
        my $pipe_tcptrace_pid = open PIPE_TCPTRACE, "$path_tcptrace -r -l -o1 -n -Q $files[$line_number] |";
        chomp(@tcptrace_data = (<PIPE_TCPTRACE>));
    }

   # my $pipe_tcptrace_pid = open PIPE_TCPTRACE, "$path_tcptrace -r -l -o1 -n -Q $files[$line_number] |"; # other possible options to consider to tcptrace: -W -u

    #chomp (my @tcptrace_data = (<PIPE_TCPTRACE>));
    close PIPE_TCPTRACE or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of tcptrace subprocess nonzero; $? $@ $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };

    my $tcptrace_start_found;
    my $tcptrace_data = '';
    for (@tcptrace_data) {
      if (not $tcptrace_start_found)
           { if (m/a->b/) { $tcptrace_start_found ++ } }
      else {
        if    (s<^\s+[^:]*:\s+(\S+)[^:]*:\s+(\S+).*><$1 $2>) { y|/| |; $tcptrace_data .= "$_ " }
        elsif ( ! m/^$/ )                                    { print OUT_FILELISTWARNINGS "$files[$line_number]\n";
                                                               warn "format error line $. fileno $line_number file $files[$line_number] LINE<$_> $! $? $@" }
      }
    }
    chop ($tcptrace_data);

    print OUT_TCPTRACECHAR "$tcptrace_data\n";
    
    if((@opt_dur)&&(!$opt_pkt)){ # duration option only
        my $pipe_tcpslice_pid = open PIPE_TCPSLICE, "$path_tcpslice $start $end  $files[$line_number] -w pipe |";
    }
    if(($opt_pkt)&&(!@opt_dur)){ # packet count option only
        my $pipe_tcpdump_pid = open PIPE_DUMP_TCPDUMP, "$path_tcpdump -c $count -r $files[$line_number] -w pipe 2>/dev/null |";
    }
    if((@opt_dur)&&($opt_pkt)){ # packet and duration option
        my $pipe_tcpslice_pid = open PIPE_TCPSLICE, "$path_tcpslice $start $end $files[$line_number] -w tmpipe |";
        my $pipe_tcpdump_pid = open PIPE_DUMP_TCPDUMP, "$path_tcpdump -c $count -r tmpipe -w pipe |";
    }
    if ((@opt_dur)||($opt_pkt)){ # options specified tcpdump
        my $pipe_tcptrace_pid = open PIPE_TCPDUMP, "$path_tcpdump -e -nn -tt -q -v -r pipe 2>/dev/null |";
    }
    if ((!@opt_dur)&&(!$opt_pkt)){ # no options specififed tcpdump
        my $pipe_tcptrace_pid = open PIPE_TCPDUMP, "$path_tcpdump -e -nn -tt -q -v -r $files[$line_number] 2>/dev/null |";
    }

#    my $pipe_tcpdump_pid = open PIPE_TCPDUMP, "$path_tcpdump -e -nn -tt -q -v -r $files[$line_number] 2>/dev/null |"; # perhaps -S; had -vv and no -q
    my $pid_mtk = open2(*PIPE_MTK_READER, *PIPE_MTK_WRITER, "$path_mtkdemo -b0 -c0.00000000000000000001 | tail -n 1" ) or warn;
    my $pid_mtk_ab = open2(*PIPE_MTK_READER_AB, *PIPE_MTK_WRITER_AB, "$path_mtkdemo -b0 -c0.00000000000000000001 | tail -n 1" ) or warn;
    my $pid_mtk_ba = open2(*PIPE_MTK_READER_BA, *PIPE_MTK_WRITER_BA, "$path_mtkdemo -b0 -c0.00000000000000000001 | tail -n 1" ) or warn;
    my $pid_fft = open2(*PIPE_FFT_READER, *PIPE_FFT_WRITER, "$path_fft -m -o 32 2>/dev/null" ) or warn;
    my $pid_fft_ab = open2(*PIPE_FFT_READER_AB, *PIPE_FFT_WRITER_AB, "$path_fft -m -o 32 2>/dev/null" ) or warn;
    my $pid_fft_ba = open2(*PIPE_FFT_READER_BA, *PIPE_FFT_WRITER_BA, "$path_fft -m -o 32 2>/dev/null" ) or warn;
    my (@tm_ip, @tm_time, @tm_data); my (@ab_tm_ip, @ab_tm_time, @ab_tm_data); my (@ba_tm_ip, @ba_tm_time, @ba_tm_data);
    for (0, 1, 2) { @{$ab_data[$_]} = @{$ba_data[$_]} = @{$all_data[$_]} = () }

    my $first_src_ip;
    my $pipe_tcpdump_line_number = 0;
    for (<PIPE_TCPDUMP>) {
      my $skip;

      if ($size > 1_000_000 and not (++$pipe_tcpdump_line_number % 1_000)) {
        my $ln = floor ($pipe_tcpdump_line_number / 1_000);
        print STDERR "\rfileno $line_number $size/$total_size brate $brate frate $frate \"$files[$line_number]\" eta $eta ln $ln      \b\b\b\b\b\b";
      }

#TCPDUMP 3.9
#1061333838.092419 00:03:a0:1a:2e:c0 > 2c:5d:f8:62:c5:00, IPv4, length 235: (tos 0x0, ttl  53, id 55246, offset 0, flags [DF], proto: TCP (6), length: 221) 18.79.2.169.33244 > 193.62.203.13.3306: tcp 181
##UDP
#1061334057.879289 00:03:a0:1a:2e:c0 > 79:b1:5a:fc:c5:00, IPv4, length 98: (tos 0x0, ttl  52, id 19606, offset 0, flags [none], proto: UDP (17), length: 84) 139.175.252.16.3621 > 193.62.197.205.53: UDP, length 56
#TCPDUMP 3.8
#1061333838.092419 00:03:a0:1a:2e:c0 > 2c:5d:f8:62:c5:00, IPv4, length 235: IP (tos 0x0, ttl  53, id 55246, offset 0, flags [DF], proto 6, length: 221) 18.79.2.169.33244 > 193.62.203.13.3306: tcp 181
#UDP
#1061334057.879289 00:03:a0:1a:2e:c0 > 79:b1:5a:fc:c5:00, IPv4, length 98: IP (tos 0x0, ttl  52, id 19606, offset 0, flags [none], proto 17, length: 84) 139.175.252.16.3621 > 193.62.197.205.53: UDP, length 56


      my ($time, $elen, $len, $src_ip, $data);
      ($time, $elen, $len, $src_ip, $data) = m<^(\d{10}\.\d{6,7}) ([^,]+, IPv4, length \d+|ip): IP \([^:]+: (\d+)\) (\d+\.\d+\.\d+\.\d+)\.\d+ \> [^:]+: tcp (\d+)$> or
      ($time, $elen, $len, $src_ip, $data) = m<^(\d{10}\.\d{6,7}) (ip): IP \([^:]+: (\d+)\) (\d+\.\d+\.\d+\.\d+)\.\d+ \> [^:]+: UDP, length (\d+)$> or
# Patch for tcpdump 3.9
      ($time, $elen, $len, $src_ip, $data) = m<^(\d{10}\.\d{6,7}) ([^,]+, IPv4, length \d+|ip): \([^:]+:[^:]+: (\d+)\) (\d+\.\d+\.\d+\.\d+)\.\d+ \> [^:]+: tcp (\d+)$> or
      ($time, $elen, $len, $src_ip, $data) = m<^(\d{10}\.\d{6,7}) (ip): \([^:]+:[^:]+:(\d+)\) (\d+\.\d+\.\d+\.\d+)\.\d+ \> [^:]+: UDP, length (\d+)$> or

        do { warn "tcpdump match fails: <$_> $! $@ <$1+$2+$3+$4+$5>, line $. for file $files[$line_number] fileno $line_number\n"; $skip++; };

      if ($skip) {
        print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      } else {
        if ($elen eq "ip") { $elen = $len + 14 } else { $elen =~ s/.*length // }
        if ($time =~ m/\.10{6}/) { $time = floor ($time) + 1; $time .= ".000000" }
        push @tm_ip, $src_ip; push @tm_time, $time; push @tm_data, $data;
        $first_src_ip ||= $src_ip;
        my @data;
        $total_frames ++;
        push @{$all_data[0]}, ($data [0] = $elen); # Wire # tcpdump -e ....
        push @{$all_data[1]}, ($data [1] = $len); # IP
        push @{$all_data[2]}, ($data [2] = $elen - $len); # Control

        for (0, 1, 2) { $stat [$_] -> add_data ($data [$_]) }

        if ($src_ip eq $first_src_ip) { # a->b
          my @data_ab;
          $total_frames_ab ++;
          push @ab_tm_ip, $src_ip; push @ab_tm_time, $time; push @ab_tm_data, $data;
          push @{$ab_data[0]}, ($data_ab [0] = $elen); # Wire
          push @{$ab_data[1]}, ($data_ab [1] = $len); # IP
          push @{$ab_data[2]}, ($data_ab [2] = $elen - $len); # Control
          for (0, 1, 2) { $stat_ab [$_] -> add_data ($data_ab [$_]) }

          if (not defined $offset_ab) { $offset_ab = $time }
          printf PIPE_MTK_WRITER_AB "%06f %d\n", ($time - $offset_ab), $elen;
        } else { # b->a
          my @data_ba;
          $total_frames_ba ++;
          push @ba_tm_ip, $src_ip; push @ba_tm_time, $time; push @ba_tm_data, $data;
          push @{$ba_data[0]}, ($data_ba [0] = $elen); # Wire
          push @{$ba_data[1]}, ($data_ba [1] = $len); # IP
          push @{$ba_data[2]}, ($data_ba [2] = $elen - $len); # Control
          for (0, 1, 2) { $stat_ba [$_] -> add_data ($data_ba [$_]) }

          if (not defined $offset_ba) { $offset_ba = $time }
          printf PIPE_MTK_WRITER_BA "%06f %d\n", ($time - $offset_ba), $elen;
        }

        if (not defined $offset) { $offset = $time }
        printf PIPE_MTK_WRITER "%06f %d\n", ($time - $offset), $elen;
      }
    }

    do { # NPU_PADDING: takes a stream of <timestamp> count and blocks them up to size size of $step outputting an entry for every $step of time
      my $step=0.001; # $step=0.0001;
      my $last = undef;
      my $y = "0";
      my $tx;

      for my $x (map { $_ - $tm_time [0] } @tm_time) {
        if(!defined($last)) { $last=$x; printf PIPE_FFT_WRITER "%.10f %f # start\n",$x,"0.0" }
        $y ++;
        if(($last + $step) <= $x) {
            $tx=$last;
            while($tx < $x) {
                $tx=$tx+$step;
                if($y == "0") { printf PIPE_FFT_WRITER "%.10f %f\n",$tx,$y; }
                else{ $y --; printf PIPE_FFT_WRITER "%.10f %f\n",$tx,$y; $y = "0"; }
            }
            $tx=$tx+$step;
            $y ++;
            printf PIPE_FFT_WRITER "%.10f %f\n",$tx,$y;
            $last=$tx+$step;
            $y="0";
        }
      }
    };

    do { # NPU_PADDING: takes a stream of <timestamp> count and blocks them up to size size of $step outputting an entry for every $step of time
      my $step=0.001; # $step=0.0001;
      my $last = undef;
      my $y = "0";
      my $tx;

      for my $x (map { $_ - $ab_tm_time [0] } @ab_tm_time) {
        if(!defined($last)) { $last=$x; printf PIPE_FFT_WRITER_AB "%.10f %f # start\n",$x,"0.0" }
        $y ++;
        if(($last + $step) <= $x) {
            $tx=$last;
            while($tx < $x) {
                $tx=$tx+$step;
                if($y == "0") { printf PIPE_FFT_WRITER_AB "%.10f %f\n",$tx,$y; }
                else{ $y --; printf PIPE_FFT_WRITER_AB "%.10f %f\n",$tx,$y; $y = "0"; }
            }
            $tx=$tx+$step;
            $y ++;
            printf PIPE_FFT_WRITER_AB "%.10f %f\n",$tx,$y;
            $last=$tx+$step;
            $y="0";
        }
      }
    };

    do { # NPU_PADDING: takes a stream of <timestamp> count and blocks them up to size size of $step outputting an entry for every $step of time
      my $step=0.001; # $step=0.0001;
      my $last = undef;
      my $y = "0";
      my $tx;

      for my $x (map { $_ - $ba_tm_time [0] } @ba_tm_time) {
        if(!defined($last)) { $last=$x; printf PIPE_FFT_WRITER_BA "%.10f %f # start\n",$x,"0.0" }
        $y ++;
        if(($last + $step) <= $x) {
            $tx=$last;
            while($tx < $x) {
                $tx=$tx+$step;
                if($y == "0") { printf PIPE_FFT_WRITER_BA "%.10f %f\n",$tx,$y; }
                else{ $y --; printf PIPE_FFT_WRITER_BA "%.10f %f\n",$tx,$y; $y = "0"; }
            }
            $tx=$tx+$step;
            $y ++;
            printf PIPE_FFT_WRITER_BA "%.10f %f\n",$tx,$y;
            $last=$tx+$step;
            $y="0";
        }
      }
    };

    close PIPE_TCPDUMP or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of tcpdump subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    close (PIPE_TCPSLICE);
    close (PIPE_TRACE_TCPDUMP); 
    close (PIPE_DUMP_TCPDUMP);
    close (PIPE_MTK_WRITER) or warn "closing mtk writer";
    close (PIPE_MTK_WRITER_AB) or warn "closing mtk writer (ab)";
    close (PIPE_MTK_WRITER_BA) or warn "closing mtk writer (ba)";
    close (PIPE_FFT_WRITER) or warn "closing fft writer";
    close (PIPE_FFT_WRITER_AB) or warn "closing fft writer (ab)";
    close (PIPE_FFT_WRITER_BA) or warn "closing fft writer (ba)";
    chomp ($_ = <PIPE_MTK_READER>); split; print OUT_EFFBANDALL "$_[3]\n";
    chomp ($_ = <PIPE_MTK_READER_AB>); split; print OUT_EFFBANDAB "$_[3]\n";
    chomp ($_ = <PIPE_MTK_READER_BA>); split; print OUT_EFFBANDBA "$_[3]\n";

    close (PIPE_MTK_READER) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of mtk subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    close (PIPE_MTK_READER_AB) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of mtk (ab) subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    close (PIPE_MTK_READER_BA) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of mtk (ba) subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    waitpid ($pid_mtk, 0);
    waitpid ($pid_mtk_ab, 0);
    waitpid ($pid_mtk_ba, 0);

    my @fft = map  { atan ($_ -> [0]) }
              sort { $b->[1] <=> $a->[1] }
              map  { do { chomp; split; $_[0] eq "fft:" ? () : [@_] } }
              <PIPE_FFT_READER>;
    @fft = (@fft, ("NA") x 10)[0..9];
    print OUT_FFTALL "@fft\n";

    my @fft_ab = map  { atan ($_ -> [0]) }
                 sort { $b->[1] <=> $a->[1] }
                 map  { do { chomp; split; $_[0] eq "fft:" ? () : [@_] } }
                 <PIPE_FFT_READER_AB>;
    @fft_ab = (@fft_ab, ("NA") x 10)[0..9];
    print OUT_FFTAB "@fft_ab\n";

    my @fft_ba = map  { atan ($_ -> [0]) }
                 sort { $b->[1] <=> $a->[1] }
                 map  { do { chomp; split; $_[0] eq "fft:" ? () : [@_] } }
                 <PIPE_FFT_READER_BA>;
    @fft_ba = (@fft_ba, ("NA") x 10)[0..9];
    print OUT_FFTBA "@fft_ba\n";

    close (PIPE_FFT_READER) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of fft subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    close (PIPE_FFT_READER_AB) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of fft (ab) subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };
    close (PIPE_FFT_READER_BA) or do {
      print OUT_FILELISTWARNINGS "$files[$line_number]\n";
      warn "exit status of fft (ba) subprocess nonzero; $?, $@, $!; line $. for file $files[$line_number] fileno $line_number DEF<$_>";
    };

    waitpid ($pid_fft, 0);
    waitpid ($pid_fft_ab, 0);
    waitpid ($pid_fft_ba, 0);

    sub do_data_stats {
      my $all_data_ref = shift;
      my @my_all_data = @$all_data_ref;
      my $stat_ref = shift;
      my @my_stat = @$stat_ref;
      my $line_number = shift;
      my $my_total_frames = shift;

      my ($location_1, $index_1, $fraction_1, $location_3, $index_3, $fraction_3);
      $location_1 = ($my_total_frames - 1)  / 4;      $location_3 = 3 * $location_1;
      $index_1 = int $location_1;               $index_3 = int $location_3;
      $fraction_1 = $location_1 - $index_1;     $fraction_3 = $location_3 - $index_3;
      my @all_stats;

      if ($my_total_frames == 1) {
        @_ = ($my_all_data [0][0], $my_all_data [1][0], $my_all_data [2][0]);
        return "$time_by_ln[$line_number] $_[0] $_[0] $_[0] $_[0] $_[0] $_[0] 0 $_[1] $_[1] $_[1] $_[1] $_[1] $_[1] 0 $_[2] $_[2] $_[2] $_[2] $_[2] $_[2] 0\n";
      } elsif ($my_total_frames == 0) {
        return "NA 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";
      }

      for my $stat_of (0, 1, 2) {
        for my $stat_type (qw( min count median mean count max variance )) {
          push @all_stats, eval "\$my_stat[$stat_of] -> $stat_type ()";
        }

        @{$my_all_data [$stat_of]} = sort { $a <=> $b } @{$my_all_data [$stat_of]};

        if ($#{$my_all_data[0]}) {
          $all_stats [$stat_of*7 + 1] = ${$my_all_data [$stat_of]}[$index_1] + (${$my_all_data [$stat_of]}[$index_1 + 1] - ${$my_all_data [$stat_of]}[$index_1]) * $fraction_1; # q1
          $all_stats [$stat_of*7 + 4] = ${$my_all_data [$stat_of]}[$index_3] + (${$my_all_data [$stat_of]}[$index_3 + 1] - ${$my_all_data [$stat_of]}[$index_3]) * $fraction_3; # q3
        } else {
          @all_stats = (($my_all_data [0] [0]) x 6, (0), ($my_all_data [1] [0]) x 6, (0), ($my_all_data [2] [0]) x 6);
        }
      }

      for (0, 1, 2) { $my_stat [$_] -> DESTROY () }
      undef (@my_stat);
      undef (@my_all_data);
      "$time_by_ln[$line_number] @all_stats\n";
    }

    print OUT_DATASTATSALL do_data_stats (\@all_data, \@stat, $line_number, $total_frames);
    print OUT_DATASTATSAB do_data_stats (\@ab_data, \@stat_ab, $line_number, $total_frames_ab);
    print OUT_DATASTATSBA do_data_stats (\@ba_data, \@stat_ba, $line_number, $total_frames_ba);

    undef (@stat);
    undef (@stat_ab);
    undef (@stat_ba);
    undef (@all_data); undef (@ab_data); undef (@ba_data);

  # we do not deal with this case: $l=length($lines[0]); if ($l > 6) { $time="NA"; $min=0; $q1=0; $med=0; $mean=0; $q3=0; $max=0; $var=0; $min_ip=0; $q1_ip=0; $med_ip=0; $mean_ip=0; $q3_ip=0; $max_ip=0; $var_ip=0; $min_control=0; $q1_control=0; $med_control=0; $mean_control=0; $q3_control=0; $max_control=0; $var_control=0; }

    {
      my $Y=0; my $B=0; my $D=0; my $duration=0; my $idle=0; my $ACK=0; my $pR="NA"; # defining the variables in use.
      my $timeS = 0; my $timeF = 0; my $timestamp = 0; my $d = 0; my $p = 0; my $t1 = 0; my $i = 0; my $t2 = 0; my $DUR = 0; my $tf = 0; my $ts = 0; my $diff = 0;

      if ($#tm_ip != 0) {
        $timestamp=$tm_time[0];
        chomp($timestamp);

        $d=$tm_data[0];

        if ($d != 0) { $pR=$tm_ip[0]; $B=1; $timeS=$tm_time[0]; }
        $t1=$tm_time[0]; $t2=$tm_time[$#tm_time]; $DUR = $t2 - $t1;

        #deals with the case when there is only one packet transfered.
        if ( $#tm_data == 0 ){
           print OUT_TRANSFERMODE "$timestamp 0 0 0 0 0 0\n";
        } else {
          for ($i = 1; $i <= $#tm_time; ++$i) {
            if ($tm_data[$i] != 0) {
               if ($B == 0) { $timeS=$tm_time[$i]; }
               $p=$tm_ip[$i];
               if ($p eq $pR) { $B=$B+1; }
               else {
                  if ($B >= 3) { $Y=$Y+1; $timeF=$tm_time[$i-$ACK-1]; $duration=$timeF-$timeS; $D=$D+$duration; }
                  $timeS=$tm_time[$i]; $B=1; $pR=$p;
               }
               $ACK=0;
            } else { $ACK=$ACK+1; }
            $ts=$tm_time[$i-1]; $tf=$tm_time[$i]; $diff=$tf-$ts;
            if ($diff > 2) { $idle=$idle+$diff }
          }

          if ($B >= 3) {
            $Y=$Y+1; $timeF=$tm_time[$i-$ACK] || 0; $duration=$timeF-$timeS; $D=$D+$duration; # the || 0 is a change
          } else { if ($Y != 0) { $Y=$Y+1 } }

          if ($DUR == 0) { $DUR = 0.001 }
          my $percent_bulk = $D/$DUR*100;
          my $percent_idle = $D/$DUR*100;
          print OUT_TRANSFERMODE "$Y $D $DUR $percent_bulk $idle $percent_idle\n";
        }
      } else { print OUT_TRANSFERMODE "0 0 0 0 0 0\n" }
    }

    sub do_time_stats {
      my $tm_time_ref = shift;
      my @my_tm_time = @$tm_time_ref;
      my $line_number = shift;
      my $stat=Statistics::Descriptive::Full->new();
      my $i=0;
      my @timedata; my $time; my $q1; my $q3; my $location1; my $location3; my $fraction1; my $fraction3; my $index1; my $index3;
      my ($min, $median, $mean, $variance, $max);
      my @lines = ($my_tm_time [0], (map { ($my_tm_time[$_] - $my_tm_time[$_-1]) } (1..$#my_tm_time)));
      if ($#my_tm_time == 0 or $#lines == 0) { return "NA 0 0 0 0 0 0 0\n" }
      elsif ($#my_tm_time == 1) { $_ = $lines [1]; return "$time_by_ln[$line_number] $_ $_ $_ $_ $_ $_ 0\n" }
      my $l=length($time = $lines[0]);

      if ($l > 6) {
        for (@lines) { $i++; if ($i != 1) { $timedata[$i-2]=$_; $stat->add_data($_) } }

        if ($#timedata==0) { $min= $q1= $median = $mean= $q3= $max=$timedata[0]; $variance=0; }
        elsif ($i==1) { $time="NA"; $min=0; $q1=0; $median=0; $mean=0; $q3=0; $max=0; $variance=0; }
        else {
          my @sorted=sort { $a <=> $b } @timedata;

          $location1=$#timedata/4; $index1=int $location1; $fraction1=$location1 - $index1;
          $location3=$location1*3; $index3=int $location3; $fraction3=$location3-$index3;
          $q1=$sorted[$index1]+($sorted[$index1+1]-$sorted[$index1])*$fraction1;
          $q3=$sorted[$index3]+($sorted[$index3+1]-$sorted[$index3])*$fraction3;

          for (qw( min median mean variance max )) { eval "\$$_ = \$stat->$_ ()" }
        }
      } else { $time="NA"; $min=0; $q1=0; $median=0; $mean=0; $q3=0; $max=0; $variance=0; }

      chomp($time,$min,$q1,$median,$mean,$q3,$max,$variance);
      $stat -> DESTROY ();
      undef (@my_tm_time);
      "$time_by_ln[$line_number] $min $q1 $median $mean $q3 $max $variance\n";
    }

    print OUT_TIMESTATSALL do_time_stats (\@tm_time, $line_number);
    print OUT_TIMESTATSAB do_time_stats (\@ab_tm_time, $line_number);
    print OUT_TIMESTATSBA do_time_stats (\@ba_tm_time, $line_number);

    undef (@tm_time); undef (@ab_tm_time); undef (@ba_tm_time);

    print OUT_FILELISTPROCESSED "$files[$line_number]\n" or die "open: $!";

    undef ($brate); undef ($frate); undef ($eta);
    undef (@stat); undef (@stat_ab); undef (@stat_ba);

    undef (@all_data); undef (@ab_data); undef (@ba_data);
    undef (@tcptrace_data);

  } else { warn "fileno $line_number \"$files[$line_number]\" is too big!"; print OUT_FILELISTSKIPPED "$files[$line_number]\n" }
  undef($size);
}

# sync; sleep
close OUT_FILELISTWARNINGS or warn;
close OUT_FILELISTSKIPPED or warn;
close OUT_FILELISTPROCESSED or warn;
close OUT_TRANSFERMODE or warn;
close OUT_FFTALL or warn;
close OUT_FFTAB or warn;
close OUT_FFTBA or warn;
close OUT_EFFBANDALL or warn;
close OUT_EFFBANDAB or warn;
close OUT_EFFBANDBA or warn;
close OUT_TCPTRACECHAR or warn;
close OUT_TIMESTATSALL or warn;
close OUT_DATASTATSALL or warn;

rm ("-fr pipe tmpipe "); # Remove the pipes as they are not needed

print STDERR "\n\nReached normal termination\n";

1;

__END__
