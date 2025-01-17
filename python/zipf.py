import numpy as np
import datetime
from pathlib import Path
from multiprocessing import Pool

def generate(n: int, a: float, max_value: int, filepath: str, batch_size: int = 1000000):
    with open(filepath, 'w') as f:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vs = np.random.zipf(a, end - start)
            vs %= max_value
            s = '\n'.join([str(int(v)) for v in vs])
            f.write(s + '\n')

def generate_task(args):
    n, a, max_value, filepath = args
    print(f"starting {a} at {datetime.datetime.now()}")
    generate(n, a, max_value, filepath)
    print(f"finished {a} at {datetime.datetime.now()}")

if __name__ == "__main__":
    folder_path = "E:/traces/zipf"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    step = 0.01
    max_a = 2.29
    pcap_size = 100000000

    tasks = [(pcap_size, np.round(a, 2), 0xFFFFFFFF, f"{folder_path}/zipf-{np.round(a, 2)}.txt")
             for a in np.arange(1+step, max_a, step)]

    with Pool(16) as pool:
        pool.map(generate_task, tasks)
