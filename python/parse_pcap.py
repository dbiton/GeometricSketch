from scapy.all import *
import socket, struct

path_pcap = "../pcaps/capture.pcap"
path_output = "../pcaps/capture.out"

def ip2long(ip):
    """
    Convert an IP string to long
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]


packets_per_print = 1e5
scapy_cap = PcapReader(path_pcap)
file = open(path_output, "w")
i = 1
for p in scapy_cap:
    if "IP" in p:
        src = str(ip2long(p["IP"].src))
        file.write(src + '\n')
        if i % packets_per_print == 0:
            print(i, "packets parsed")
        i += 1

