import numpy as np
import matplotlib.pyplot as plt
import subprocess
import ctypes

tester_path = "build/DynamicSketch"
pcap_path = "pcaps/file.out"
args = f"--file {pcap_path} --type dynamic --repeat 100 log_accuracy"
subprocess.call(f"{tester_path} {args}", shell=True)

def plot_ip_distribution(filepath):
    file = open(filepath, "r")
    data = np.array([int(ip) for ip in file.readlines()])
    uint32_max = 0xffffffff
    bin_count = 100
    bins = np.arange(0, uint32_max, uint32_max / bin_count)
    counts, bins = np.histogram(data, bins=bins)
    plt.stairs(counts, bins)
    plt.show()

plot_ip_distribution("C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\pcaps\\large.out")