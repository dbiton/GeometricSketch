from scapy.all import *
import socket, struct


def ip2long(ip):
    """
    Convert an IP string to long
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]


scapy_cap = PcapReader('file2.pcap')
file = open("capture.txt", "a")
for p in scapy_cap:
    if IP in p:
        src = str(ip2long(p[IP].src))
        dst = str(ip2long(p[IP].dst))
        print(src)
        print(dst)
        file.write(src + "\n" + dst + "\n")
