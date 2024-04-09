import zipf
import os

PCAP_SIZE = 37700000
ZIPFIAN_PARAMETER = 1.02
MAX_KEY = 0xFFFFFFFF
if os.name == 'nt':
    TRACE_PATH = f"..\\pcaps\\capture.txt"
else:
    TRACE_PATH = f"../pcaps/capture.txt"


if __name__ == "__main__":
    zipf.generate(PCAP_SIZE, ZIPFIAN_PARAMETER, MAX_KEY, TRACE_PATH)