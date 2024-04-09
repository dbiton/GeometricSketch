import numpy as np
import datetime
from pathlib import Path


def generate(n: int, a: float, max_value: int, filepath: str):
    vs = np.random.zipf(a, n)
    vs %= max_value
    with open(filepath, 'w') as f:
        s = '\n'.join([str(int(v)) for v in vs])
        f.write(s)


if __name__ == "__main__":
    folder_path = "../pcaps/zipf"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    step = 0.01
    max_a = 2.29
    pcap_size = 1000000
    for a in np.arange(2.2, max_a, step):
        a_rounded = np.round(a, 2)
        print("starting", a_rounded, "at", datetime.datetime.now())
        generate(pcap_size, a_rounded, 0xFFFFFFFF, f"{folder_path}/zipf-{a_rounded}.txt")
