### 以流为单位，对于流的长度(依据packet的个数)进行filter
#### 方向(服务器的端口为443)
- client--->server:dport=443
- server--->client:sport=443

#### Forward packets
- Forward total Bytes （数值，总的长度）
- Min forward inter arrival time difference （时间信息）
- Max forward inter arrival time difference
- Mean forward inter arrival time difference
- STD forward inter arrival time difference
- Mean forward packets
- STD forward packets

#### Backward packets
- Backward total Bytes
- Min backward inter arrival time difference
- Max backward inter arrival time difference
- Mean backward inter arrival time difference
- STD backward inter arrival time difference
- Mean backward packets
- STD backward packets
- Mean forward TTL value
- Minimum forward packet
- Minimum backward packet
- Maximum forward packet
- Maximum backward packet

#### Total packets
- Minimum packet size
- Maximum packet size
- Mean packet size
- Packet size variance
- TCP initial window size
- TCP window scaling factor
- SSL compression methods
- SSL extension count
- SSL extension: SNI(数值，表示为向量形式，需要乘以权重)
- SSL chiper methods（类别，需要进行one-hot编码）
- SSL session ID len (数值，session)
- Forward peak MAX throughput
- Mean throughput of backward peaks（数值）
- Max throughput of backward peaks
- Backward min peak throughput
- Backward STD peak throughput
- Forward number of bursts
- Backward number of bursts
- Forward min peak throughput
- Mean throughput of forward peaks
- Forward STD peak throughput
- Mean backward peak inter arrival time diffd scalable feature selection approach for internet tra
- Minimum backward peak inter arrival time diff
- Maximum backward peak inter arrival time diff
- STD backward peak inter arrival time diff
- Mean forward peak inter arrival time diff
- Minimum forward peak inter arrival time diff
- Maximum forward peak inter arrival time diff
- STD forward peak inter arrival time diff

#### Keep alive packets
- TCP Maxiumu Segment Size
- Forward SSL Version

#### 其他特征
- shanno
- entropy
- Fourier-transform of packet






| 序号 | 字段名 | 描述 |
|------|--------|------|
| 1 | Server Port | Port Number at server; we can establish server and client ports as we limit ourselves to flows for which we see the initial connection set-up.|
| 2 | Client Port | Port Number at client
| 3 | min IAT | Minimum packet inter-arrival time for all packets of the flow (considering both directions). |
| 4 | q1 | IAT First quartile inter-arrival time |
| 5 | med IAT | Median inter-arrival time |
| 6 | mean IAT | Mean inter-arrival time |
| 7 | q3 IAT | Third quartile packet inter-arrival time |
| 8 | max IAT | Maximum packet inter-arrival time |
| 9 | var IAT | Variance in packet inter-arrival time |
| 10 | min data wire | Minimum of bytes in (Ethernet) packet, using the size of the packet on the wire. |
| 11 | q1 data wire | First quartile of bytes in (Ethernet) packet |
| 12 | med data wire | Median of bytes in (Ethernet) packet |
| 13 | mean data wire | Mean of bytes in (Ethernet) packet |
| 14 | q3 data wire | Third quartile of bytes in (Ethernet) packet |
| 15 | max data wire | Maximum of bytes in (Ethernet) packet |
| 16 | var data wire | Variance of bytes in (Ethernet) packet |
| 17 | min data ip | Minimum of total bytes in IP packet, using the size of payload declared by the IP packet |
| 18 | q1 data ip | First quartile of total bytes in IP packet |
| 19 | med data ip | Median of total bytes in IP packet | 
| 20 | mean data ip | Mean of total bytes in IP packet |
| 21 | q3 data ip | Third quartile of total bytes in IP packet
| 22 | max data ip | Maximum of total bytes in IP packet
| 23 | var data ip | Variance of total bytes in IP packet
| 24 | min data control | Minimum of control bytes in packet, size of the (IP/TCP) packet header |
| 25 | q1 data control | First quartile of control bytes in packet |
| 26 | med data control | Median of control bytes in packet |
| 27 | mean data control | Mean of control bytes in packet |
| 28 | q3 data control | Third quartile of control bytes in packet |
| 29 | max data control | Maximum of control bytes in packet |
| 30 | var data control | Variance of control bytes packet |
| 31 | total packets a b | The total number of packets seen (client→server). |
| 32 | total packets b a | ” (server→client) |
| 33 | ack pkts sent a b | The total number of ack packets seen (TCP segments seen with the ACK bit set) (client→server).
| 34 | ack pkts sent b a | ” (server→client)
| 35 | pure acks sent a b | The total number of ack packets seen that were not piggy-backed with data (just the TCP header and no TCP data payload) and did not have any of the SYN/FIN/RST flags set(client→server)
| 36 | pure acks sent b a | ” (server→client)
| 37 | sack pkts sent a b | The total number of ack packets seen carrying TCP SACK [6] blocks (client→server)
| 38 | sack pkts sent b a | ” (server→client)
| 39 | dsack pkts sent a b | The total number of sack packets seen that carried duplicate SACK (D-SACK) [7] blocks. (client→server)
| 40 | dsack pkts sent b a | ” (server→client)
| 41 | max sack blks/ack a b | The maximum number of sack blocks seen in any sack packet. (client→server)
| 42 | max sack blks/ack b a | ” (server→client)
| 43 | unique bytes sent a b | The number of unique bytes sent, i.e., the total bytes of data sent excluding retransmitted bytes and any bytes sent doing window probing. (client→server)
| 44 | unique bytes sent b a | ” (server→client)
| 45 | actual data pkts a b | The count of all the packets with at least a byte of TCP data payload. (client→server)
| 46 | actual data pkts b a | ” (server→client)
| 47 | actual data bytes a b | The total bytes of data seen. Note that this includes bytes from retransmissions / window probe packets if any. (client→server)
| 48 | actual data bytes b a | ” (server→client)
| 49 | rexmt data pkts a b | The count of all the packets found to be retransmissions. (client→server)
| 50 | rexmt data pkts b a | ” (server→client)
| 51 | rexmt data bytes a b | The total bytes of data found in the retransmitted packets. (client→server)
| 52 | rexmt data bytes b a | ” (server→client)
| 53 | zwnd probe pkts a b | The count of all the window probe packets seen. (Window probe packets are typically sent by a sender when the receiver last advertised a zero receive window, to see if the window has opened up now). (client→server)
| 54 | zwnd probe pkts b a | ” (server→client)
| 55 | zwnd probe bytes a b | The total bytes of data sent in the window probe packets. (client→server)
| 56 | zwnd probe bytes b a | ” (server→client)
| 57 | outoforder pkts a b | The count of all the packets that were seen to arrive out of order. (client→server)
| 58 | outoforder pkts b a | ” (server→client)
| 59 | pushed data pkts a b | The count of all the packets seen with the PUSH bit set in the TCP header. (client→server)
| 60 | pushed data pkts b a | ” (server→client)
| 61 | SYN pkts sent a b | The count of all the packets seen with the SYN bits set in the TCP header respectively (client→server)
| 62 | FIN pkts sent a b | The count of all the packets seen with the FIN bits set in the TCP header respectively (client→server)
| 63 | SYN pkts sent b a | The count of all the packets seen with the SYN bits set in the TCP header respectively (server→client)
| 64 | FIN pkts sent b a | The count of all the packets seen with the FIN bits set in the TCP header respectively (server→client)
| 65 | req 1323 ws a b | If the endpoint requested Window Scaling/Time Stamp options as specified in RFC 1323[8] a ‘Y’ is printed on the respective field. If the option was not requested, an ‘N’ is printed. For example, an “N/Y” in this field means that the window-scaling option was not specified, while the Time-stamp option was specified in the SYN segment. (client→server)
| 66 | req 1323 ts a b | ...
| 67 | req 1323 ws b a | If the endpoint requested Window Scaling/Time Stamp options as specified in RFC 1323[8] a ‘Y’ is printed on the respective field. If the option was not requested, an ‘N’ is printed. For example, an “N/Y” in this field means that the window-scaling option was not specified, while the Time-stamp option was specified in the SYN segment. (client→server)
| 68 | req 1323 ts b a | ...
| 69 | adv wind scale a b | The window scaling factor used. Again, this field is valid only if the connection was captured fully to include the SYN packets. Since the connection would use window scaling if and only if both sides requested window scaling [8], this field is reset to 0 (even if a window scale was requested in the SYN packet for this direction), if the SYN packet in the reverse direction did not carry the window scale option. (client→server)
| 70 | adv wind scale b a | ” (server→client)
| 71 | req sack a b | If the end-point sent a SACK permitted option in the SYN packet opening the connection, a ‘Y’ is printed; otherwise ‘N’ is printed. (client→server)
| 72 | req sack b a | ” (server→client)
| 73 | sacks sent a b | The total number of ACK packets seen carrying SACK information. (client→server)
| 74 | sacks sent b a | ” (server→client)
| 75 | urgent data pkts a b | The total number of packets with the URG bit turned on in the TCP header. (client→server)
| 76 | urgent data pkts b a | ” (server→client)
| 77 | urgent data bytes a b | The total bytes of urgent data sent. This field is calculated by summing the urgent pointer offset values found in packets having the URG bit set in the TCP header. (client→server)
| 78 | urgent data bytes b a | ” (server→client)
| 79 | mss requested a b | The Maximum Segment Size (MSS) requested as a TCP option in the SYN packet opening the connection. (client→server)
| 80 | mss requested b a | ” (server→client)
| 81 | max segm size a b | The maximum segment size observed during the lifetime of the connection. (client→server)
| 82 | max segm size b a | ” (server→client)
| 83 | min segm size a b | The minimum segment size observed during the lifetime of the connection. (client→server)
| 84 | min segm size b a | ” (server→client)
| 85 | avg segm size a b | The average segment size observed during the lifetime of the connection calculated as the value reported in the actual data bytes field divided by the actual data pkts reported. (client→server)
| 86 | avg segm size b a | ” (server→client)
| 87 | max win adv a b | The maximum window advertisement seen. If the connection is using window scaling (both sides negotiated window scaling during the opening of the connection), this is the maximum window-scaled advertisement seen in the connection. For a connection using window scaling, both the SYN segments opening the connection have to be captured in the dumpfile for
this and the following window statistics to be accurate.(client→server)
| 88 | max win adv b a | ” (server→client)
| 89 | min win adv a b | The minimum window advertisement seen. This is the minimum window-scaled advertisement seen if both sides negotiated window scaling. (client→server)
| 90 | min win adv b a | ” (server→client)
| 91 | zero win adv a b | The number of times a zero receive window was advertised. (client→server)
| 92 | zero win adv b a | ” (server→client)
| 93 | avg win adv a b | The average window advertisement seen, calculated as the sum of all window advertisements divided by the total number of packets seen. If the connection endpoints negotiated window scaling, this average is calculated as the sum of all window-scaled advertisements divided by the number of window-scaled packets seen. Note that in the window-scaled case, the window advertisements in the SYN packets are excluded since the SYN packets themselves cannot have their window advertisements scaled, as per RFC 1323 [8]. (client→server)
| 94 | avg win adv b a | ” (server→client)
| 95 | initial window-bytes a b | The total number of bytes sent in the initial window i.e., the number of bytes seen in the initial flight of data before receiving the first ack packet from the other endpoint. Note that the ack packet from the other endpoint is the first ack acknowledging some data (the ACKs part of the 3-way handshake do not count), and any retransmitted packets in this stage are excluded. (client→server)
| 96 | initial window-bytes b a | ” (server→client)
| 97 | initial window-packets a b | The total number of segments (packets) sent in the initial window as explained above. (client→server)
| 98 | initial window-packets b a | ” (server→client)
| 99 | ttl stream length a b | The Theoretical Stream Length. This is calculated as the difference between the sequence numbers of the SYN and FIN packets, giving the length of the data stream seen. Note that this calculation is aware of sequence space wrap-arounds, and is printed only if the connection was complete (both the SYN and FIN packets were seen). (client→server)
| 100 | ttl stream length b a | ” (server→client)
| 101 | missed data a b | The missed data, calculated as the difference between the ttl stream length and unique bytes sent. If the connection was not complete, this calculation is invalid and an “NA” (Not Available) is printed. (client→server)
| 102 | missed data b a | ” (server→client)
| 103 | truncated data a b | The truncated data, calculated as the total bytes of data truncated during packet capture. For example, with tcpdump, the snaplen option can be set to 64 (with -s option) so that just the headers of the packet (assuming there are no options) are captured, truncating most of the packet data. In an Ethernet with maximum segment size of 1500 bytes, this would amount to truncated data of 1500 64 = 1436bytes for a packet.(client→server)
| 104 | truncated data b a | ” (server→client)
| 105 | truncated packets a b | The total number of packets truncated as explained above. (client→server)
| 106 | truncated packets b a | ” (server→client)
| 107 | data xmit time a b | Total data transmit time, calculated as the difference between the times of capture of the first and last packets carrying non-zero TCP data payload. (client→server)
| 108 | data xmit time b a | ” (server→client)
| 109 | idletime max a b | Maximum idle time, calculated as the maximum time between consecutive packets seen in the direction.(client→server)
| 110 | idletime max b a | ” (server→client)
| 111 | throughput a b | The average throughput calculated as the unique bytes sent divided by the elapsed time i.e., the value reported in the unique bytes sent field divided by the elapsed time (the time difference between the capture of the first and last packets in the direction). (client→server)
| 112 | throughput b a | ” (server→client)
| 113 | RTT samples a b | The total number of Round-Trip Time (RTT) samples found. tcptrace is pretty smart about choosing only valid RTT samples. An RTT sample is found only if an ack packet is received from the other endpoint for a previously transmitted packet such that the acknowledgment value is 1 greater than the last sequence number of the packet. Further, it is required that the packet being acknowledged was not retransmitted, and that no packets that came before it in the sequence space were retransmitted after the packet was transmitted. Note : The former condition invalidates RTT samples due to the retransmission ambiguity problem, and the latter condition invalidates RTT samples since it could be the case that the ack packet could be cumulatively acknowledging the retransmitted packet, and not necessarily ack-ing the packet in question. (client→server)
| 114 | RTT samples b a | ” (server→client)
| 115 | RTT min a b | The minimum RTT sample seen. (client→server)
| 116 | RTT min b a | ” (server→client)
| 117 | RTT max a b | The maximum RTT sample seen. (client→server)
| 118 | RTT max b a | ” (server→client)
| 119 | RTT avg a b | The average value of RTT found, calculated straightforward-ly as the sum of all the RTT values found divided by the total number of RTT samples. (client→server)
| 120 | RTT avg b a | ” (server→client)
| 121 | RTT stdv a b | The standard deviation of the RTT samples. (client→server)
| 122 | RTT stdv b a | ” (server→client)
| 123 | RTT from 3WHS a b | The RTT value calculated from the TCP 3-Way Hand-Shake (connection opening) [9], assuming that the SYN packets of the connection were captured. (client→server)
| 124 | RTT from 3WHS b a | ” (server→client)
| 125 | RTT full sz smpls a b | The total number of full-size RTT samples, calculated from the RTT samples of full-size segments. Full-size segments are defined to be the segments of the largest size seen in the connection. (client→server)
| 126 | RTT full sz smpls b a | ” (server→client)
| 127 | RTT full sz min a b | The minimum full-size RTT sample. (client→server)
| 128 | RTT full sz min b a | ” (server→client)
| 129 RTT full sz max a b | The maximum full-size RTT sample. (client→server)
| 130 RTT full sz max b a | ” (server→client)
| 131 RTT full sz avg a b | The average full-size RTT sample. (client→server)
| 132 RTT full sz avg b a | ” (server→client)
| 133 RTT full sz stdev a b | The standard deviation of full-size RTT samples.
(client→server)
| 134 | RTT full sz stdev b a | ” (server→client)
| 135 | post-loss acks a b | The total number of ack packets received after losses were detected and a retransmission occurred. More precisely, a post-loss ack is found to occur when an ack packet acknowledges a packet sent (acknowledgment value in the ack pkt is 1 greater than the packet’s last sequence number), and at least one packet occurring before the packet acknowledged, was retransmitted later. In other words, the ack packet is received after we observed a (perceived) loss event and are recovering from it. (client→server)
| 136 | post-loss acks b a | ” (server→client)
| 137 | segs cum acked a b | The count of the number of segments that were cumulatively acknowledged and not directly acknowledged. (client→server)
| 138 | segs cum acked b a | ” (server→client)
| 139 | duplicate acks a b | The total number of duplicate acknowledgments received. (client→server)
| 140 | duplicate acks b a | ” (server→client)
| 141 | triple dupacks a b | The total number of triple duplicate acknowledgments received (three duplicate acknowledgments acknowledging the same segment), a condition commonly used to trigger the fast-retransmit/fast-recovery phase of TCP. (client→server)
| 142 | triple dupacks b a | ” (server→client)
| 143 | max # retrans a b | The maximum number of retransmissions seen for any segment during the lifetime of the connection.(client→server)
| 144 | max # retrans b a | ” (server→client)
| 145 | min retr time a b | The minimum time seen between any two (re)transmissions of a segment amongst all the retransmissions seen. (client→server)
| 146 | min retr time b a | ” (server→client)
| 147 | max retr time a b | The maximum time seen between any two (re)transmissions of a segment. (client→server)
| 148 | max retr time b a | ” (server→client)
| 149 | avg retr time a b | The average time seen between any two (re)transmissions of a segment calculated from all the retransmissions. (client→server)
| 150 | avg retr time b a | ” (server→client)
| 151 | sdv retr time a b | The standard deviation of the retransmission-time samples obtained from all the retransmissions.(client→server)
| 152 | sdv retr time b a | ” (server→client)
| 153 | min data wire a b | Minimum number of bytes in (Ethernet) packet(client→server)
| 154 | q1 data wire a b | First quartile of bytes in (Ethernet) packet
| 155 | med data wire a b | Median of bytes in (Ethernet) packet
| 156 | mean data wire a b | Mean of bytes in (Ethernet) packet
| 157 | q3 data wire a b | Third quartile of bytes in (Ethernet) packet
| 158 | max data wire a b | Maximum of bytes in (Ethernet) packet
| 159 | var data wire a b | Variance of bytes in (Ethernet) packet
| 160 | min data ip a b | Minimum number of total bytes in IP packet
| 161 | q1 data ip a b | First quartile of total bytes in IP packet
| 162 | med data ip a b | Median of total bytes in IP packet
| 163 | mean data ip a b | Mean of total bytes in IP packet
| 164 | q3 data ip a b | Third quartile of total bytes in IP packet
| 165 | max data ip a b | Maximum of total bytes in IP packet
| 166 | var data ip a b | Variance of total bytes in IP packet
| 167 | min data control a b | Minimum of control bytes in packet
| 168 | q1 data control a b | First quartile of control bytes in packet
| 169 | med data control a b | Median of control bytes in packet
| 170 | mean data control a b | Mean of control bytes in packet
| 171 | q3 data control a b | Third quartile of control bytes in packet
| 172 | max data control a b | Maximum of control bytes in packet
| 173 | var data control a b | Variance of control bytes packet
| 174 | min data wire b a | Minimum number of bytes in (Ethernet) packet (server→client)
| 175 | q1 data wire b a | First quartile of bytes in (Ethernet) packet
| 176 | med data wire b a | Median of bytes in (Ethernet) packet
| 177 | mean data wire b a | Mean of bytes in (Ethernet) packet
| 178 | q3 data wire b a | Third quartile of bytes in (Ethernet) packet
| 179 | max data wire b a | Maximum of bytes in (Ethernet) packet
| 180 | var data wire b a | Variance of bytes in (Ethernet) packet
| 181 | min data ip b a | Minimum number of total bytes in IP packet
| 182 | q1 data ip b a | First quartile of total bytes in IP packet
| 183 | med data ip b a | Median of total bytes in IP packet
| 184 | mean data ip b a | Mean of total bytes in IP packet
| 185 | q3 data ip b a | Third quartile of total bytes in IP packet
| 186 | max data ip b a | Maximum of total bytes in IP packet
| 187 | var data ip b a | Variance of total bytes in IP packet
| 188 | min data control b a | Minimum of control bytes in packet
| 189 | q1 data control b a | First quartile of control bytes in packet
| 190 | med data control b a | Median of control bytes in packet
| 191 | mean data control b a | Mean of control bytes in packet
| 192 | q3 data control b a | Third quartile of control bytes in packet
| 193 | max data control b a | Maximum of control bytes in packet
| 194 | var data control b a | Variance of control bytes packet
| 195 | min IAT a b | Minimum of packet inter-arrival time (client→server)
| 196 | q1 IAT a b | First quartile of packet inter-arrival time
| 197 | med IAT a b | Median of packet inter-arrival time
| 198 |mean IAT a b | Mean of packet inter-arrival time
| 199 | q3 IAT a b | Third quartile of packet inter-arrival time
| 200 | max IAT a b | Maximum of packet inter-arrival time
| 201 | var IAT a b | Variance of packet inter-arrival time
| 202 | min IAT b a | Minimum of packet inter-arrival time (server→client)
| 203 | q1 IAT b a | First quartile of packet inter-arrival time
| 204 | med IAT b a | Median of packet inter-arrival time
| 205 | mean IAT b a | Mean of packet inter-arrival time
| 206 | q3 IAT b a | Third quartile of packet inter-arrival time
| 207 | max IAT b a | Maximum of packet inter-arrival time
| 208 | var IAT b a | Variance of packet inter-arrival time
| 209 | Time since last connection | Time since the last connection between these hosts
| 210 | No. transitions bulk/trans | The number of transitions between transaction mode and bulk transfer mode, where bulk transfer mode is defined as the time when there are more than three
successive packets in the same direction without any packets carrying data in the other direction 211 Time spent in bulk Amount of time spent in bulk transfer mode
| 212 Duration Connection duration
| 213 | % bulk | Percent of time spent in bulk transfer
| 214 | Time spent idle | The time spent idle (where idle time is the accumulation of all periods of 2 seconds or greater when no packet was seen in either direction)
| 215 | % idle | Percent of time spent idle
| 216 | Effective Bandwidth | Effective Bandwidth based upon entropy [10] (both directions)
| 217 | Effective Bandwidth a b | ” (client→server)
| 218 | Effective Bandwidth b a |” (server→client)
| 219 | FFT all | FFT of packet IAT (arctan of the top-ten frequencies ranked by the magnitude of their contribution) (all traffic) (Frequency #1)
| 220 | FFT all | ” (Frequency #2)
| 221 | FFT all | ” ...
| 222 | FFT all | ” ...
| 223 | FFT all | ” ...
| 224 | FFT all | ” ...
| 225 | FFT all | ” ...
| 226 | FFT all | ” ...
| 227 | FFT all | ” ...
| 228 | FFT all | ” (Frequency #10)
| 229 | FFT a b | FFT of packet IAT (arctan of the top-ten frequencies ranked by the magnitude of their contribution) (client→server) (Frequency #1)
| 230 | FFT a b | ” (Frequency #2)
| 231 | FFT a b | ” ...
| 232 | FFT a b | ” ...
| 233 | FFT a b | ” ...
| 234 | FFT a b | ” ...
| 235 | FFT a b | ” ...
| 236 | FFT a b | ” ...
| 237 | FFT a b | ” ...
| 238 | FFT b a | ” (Frequency #10)
| 239 | FFT b a | FFT of packet IAT (arctan of the top-ten frequencies ranked by the magnitude of their contribution)(server→client) (Frequency #1)
| 240 | FFT b a | ” (Frequency #2)
| 241 | FFT b a | ” ...
| 242 | FFT b a | ” ...
| 243 | FFT b a | ” ...
| 244 | FFT b a | ” ...
| 245 | FFT b a | ” ...
| 246 | FFT b a | ” ...
| 247 | FFT b a | ” ...
| 248 | FFT b a | ” (Frequency #10)
