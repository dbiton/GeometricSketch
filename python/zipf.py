import numpy as np
import datetime


def generate(n: int, a: float, max_value: int, filepath: str):
    vs = np.random.zipf(a, n)
    vs %= max_value
    with open(filepath, 'w') as f:
        s = '\n'.join([str(int(v)) for v in vs])
        f.write(s)


step = 0.01
max_a = 5.0
pcap_size = 1000000
for a in np.arange(1+step, max_a, step):
    a_rounded = np.round(a, 2)
    print("starting", a_rounded, "at", datetime.datetime.now())
    generate(pcap_size, a_rounded, 0xFFFFFFFF, f"../pcaps/zipf/zipf-{a_rounded}.txt")
