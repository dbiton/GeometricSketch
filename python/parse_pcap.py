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


scapy_cap = PcapReader(path_pcap)
source_ips = [str(ip2long(p["IP"].src)) for p in scapy_cap if "IP" in p]
with open(path_output, "w") as file:
    file.write('\n'.join(source_ips))
