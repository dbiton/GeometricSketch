import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import json
import pandas as pd

filepath_packets = "C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\pcaps\\file.out"
filepath_executable = "C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\cpp\\out\\build\\x64-Debug\\DynamicSketch.exe"


def execute_command(command: list):
    result = sp.run([filepath_executable, "--file", filepath_packets] + command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise ValueError(f"command: {' '.join(command)} caused error: {result.stderr}")
    output_dict = [json.loads(line) for line in result.stdout.split()]
    return pd.DataFrame(output_dict)



def get_packets(filepath_packets: str):
    file_packets = open(filepath_packets, "r")
    return np.array([int(ip) for ip in file_packets.readlines()])


def plot_ip_distribution(packets: np.ndarray):
    uint32_max = 0xffffffff
    bin_count = 100
    bins = np.arange(0, uint32_max, uint32_max / bin_count)
    counts, bins = np.histogram(packets, bins=bins)
    plt.stairs(counts, bins)
    plt.show()


def plot_mae_countmin_and_countsketch():
    result_countmin = execute_command(["--type", "countmin", "--repeat", "100", "log_error"])
    result_countsketch = execute_command(["--type", "countsketch", "--repeat", "100", "log_error"])
    x_countmin = np.array(result_countmin['index'].to_numpy())
    y_countmin = np.array(result_countmin['log_error'].to_numpy())
    x_countsketch = np.array(result_countsketch['index'].to_numpy())
    y_countsketch = np.array(result_countsketch['log_error'].to_numpy())
    plt.plot(x_countmin, y_countmin, label="countmin")
    plt.plot(x_countsketch, y_countsketch, label="countskech")
    plt.ylabel('error')
    plt.xlabel('packets')
    plt.legend()
    plt.grid()
    plt.show()


def plot_mae_dynamic_and_countmin(num_sketches: int, packets: list):
    pass


if __name__ == "__main__":
    packets = get_packets("C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\pcaps\\file.out")
    #plot_ip_distribution(packets)
    plot_mae_countmin_and_countsketch()
