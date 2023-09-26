import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import json
import pandas as pd

filepath_packets = "C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\pcaps\\capture.txt"
filepath_executable = "C:\\Users\\USER2\\Desktop\\projects\\DynamicSketch\\cpp\\x64\\Release\\DynamicSketch.exe"

COUNT_PACKETS_MAX = 32000000
COUNT_PACKETS = min(1000000, COUNT_PACKETS_MAX)


def execute_command(arguments: list):
    command = [filepath_executable, "--limit_file", filepath_packets, str(COUNT_PACKETS)] + arguments
    print("executing:", ' '.join(command))
    result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise ValueError(f"command: {' '.join(command)} caused error: {result.stderr}")
    output_dict = [json.loads(line) for line in result.stdout.split()]
    output_df = pd.DataFrame(output_dict).groupby('index').mean().fillna(method='ffill')
    return output_df


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
    result_countmin = execute_command(["--type", "countmin", "--repeat", "log_mean_absolute_error", "1000"])
    result_countsketch = execute_command(["--type", "countsketch", "--repeat", "log_mean_absolute_error", "1000"])
    x_countmin = np.array(result_countmin.index.to_numpy())
    y_countmin = np.array(result_countmin['log_mean_absolute_error'].to_numpy())
    x_countsketch = np.array(result_countsketch.index.to_numpy())
    y_countsketch = np.array(result_countsketch['log_mean_absolute_error'].to_numpy())
    plt.plot(x_countmin, y_countmin, label="countmin")
    plt.plot(x_countsketch, y_countsketch, label="countsketch")
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.legend()
    plt.grid()
    plt.show()


def plot_mae_dynamic_and_countmin(count_sketch: int, count_log: int):
    packets_per_modify_size = COUNT_PACKETS // (2 * count_sketch - 1)
    packets_per_log = COUNT_PACKETS // count_log

    for i in range(count_sketch):
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(28 * (i + 1)),
            "--depth", "3",
            "--repeat", "log_mean_absolute_error", str(packets_per_log),
        ])
        x_countmin = np.array(result_countmin.index.to_numpy())
        y_countmin = np.array(result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_dynamic_countmin")
        plt.plot(x_countmin, y_countmin, label=f"count-min {i + 1}")
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.plot(x_countmin, np.gradient(y_countmin, packets_per_log), label=f"count-min {i + 1}")

    result_dynamic = execute_command([
        "--type", "dynamic",
        "--width", "28",
        "--depth", "3",
        "--repeat", "modify_size_cyclic", str(packets_per_modify_size), str(count_sketch),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_dynamic = np.array(result_dynamic.index.to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())

    for xc in range(packets_per_modify_size, packets_per_modify_size * count_sketch, packets_per_modify_size):
        plt.figure(num="mae_dynamic_countmin")
        plt.axvline(x=xc, color='r', linestyle='dashed')
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.axvline(x=xc, color='r', linestyle='dashed')

    for xc in range(packets_per_modify_size * count_sketch, COUNT_PACKETS, packets_per_modify_size):
        plt.figure(num="mae_dynamic_countmin")
        plt.axvline(x=xc, color='g', linestyle='dashed')
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.axvline(x=xc, color='g', linestyle='dashed')

    plt.figure(num="mae_dynamic_countmin")
    plt.legend()
    plt.grid()
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.savefig('mae_dynamic_countmin.png')

    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.legend()
    plt.grid()
    plt.plot(x_dynamic, np.gradient(y_dynamic, packets_per_log), label="dynamic")
    plt.ylabel("MAE'")
    plt.xlabel('packets')
    plt.savefig('mae_dynamic_countmin_derivative.png')

    plt.show()


def plot_operations_per_second(dynamic_max_size: int):
    packets_per_expand = COUNT_PACKETS // dynamic_max_size

    result = execute_command([
        "--type", "dynamic",
        "--repeat", "log_update_time", str(packets_per_expand),
        "--repeat", "log_query_time", str(packets_per_expand),
        "--repeat", "expand", str(packets_per_expand)
    ])

    x = np.arange(1, dynamic_max_size+1)
    y_update = np.array(result['log_update_time'].to_numpy())
    y_query = np.array(result['log_query_time'].to_numpy())

    plt.plot(x, 1 / np.array(y_update), label="update")
    plt.plot(x, 1 / np.array(y_query), label="query")
    plt.ylabel('Operations/second')
    plt.xlabel('Dynamic sketch size')
    plt.legend()
    plt.grid()
    plt.savefig('operations_per_second.png')
    plt.show()


if __name__ == "__main__":
    #plot_mae_dynamic_and_countmin(3, 128)
    plot_operations_per_second(64)
