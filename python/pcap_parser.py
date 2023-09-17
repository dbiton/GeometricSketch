from scapy.all import *
import socket, struct

path_pcap = "pcaps/file.pcap"
path_output = "pcaps/file.out"

def ip2long(ip):
    """
    Convert an IP string to long
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]


scapy_cap = PcapReader(path_pcap)
file = open(path_output, "w")
for p in scapy_cap:
    if "IP" in p:
        src = str(ip2long(p["IP"].src))
        dst = str(ip2long(p["IP"].dst))
        file.write(src + '\n')
