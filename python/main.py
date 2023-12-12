import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import subprocess as sp
import json
import pandas as pd


filepath_packets = "/home/dbiton/Desktop/Projects/DynamicSketch/pcaps/capture.txt"
filepath_executable = "../cpp/build-DynamicSketch-Desktop-Debug/DynamicSketch"

COUNT_PACKETS_MAX = 32000000
COUNT_PACKETS = min(1000000, COUNT_PACKETS_MAX)


def execute_command(arguments: list):
    command = [filepath_executable, "--limit_file", filepath_packets, str(COUNT_PACKETS)] + arguments
    print("executing:", ' '.join(command))
    result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise ValueError(f"command: {' '.join(command)} caused error: {result.stderr}")
    raw_output = result.stdout.replace("\n", "").replace("\t", "")[:-2] + ']'
    output_dict = json.loads(raw_output)
    output_df = pd.DataFrame(output_dict)#.groupby('index').mean().fillna(method='ffill')
    return output_df


def get_packets(filepath_packets: str):
    file_packets = open(filepath_packets, "r")
    return np.array([int(ip) for ip in file_packets.readlines()])


def plot_ip_distribution():
    packets = get_packets(filepath_packets)
    uint32_max = 0xffffffff
    bin_count = 2048*4
    bins = np.arange(0, uint32_max, uint32_max / bin_count)
    counts, bins = np.histogram(packets, bins=bins)
    plt.ylabel('packet count')
    plt.xlabel('ip as integer')
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
    plt.title('packets distribution')
    plt.legend()
    plt.grid()
    plt.show()


def plot_mae_dynamic_and_countmin(count_sketch: int, count_log: int):
    packets_per_modify_size = COUNT_PACKETS // (2 * count_sketch - 1)
    packets_per_log = COUNT_PACKETS // count_log

    for i in range(count_sketch):
        result_countmin = execute_command([
            "--type", "elastic",
            "--width", str(28 * (i + 1)),
            "--depth", "300",
            "--repeat", "log_mean_absolute_error", str(packets_per_log),
        ])
        x_countmin = np.array(result_countmin['index'].to_numpy())
        y_countmin = np.array(result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_dynamic_countmin")
        plt.plot(x_countmin, y_countmin, label=f"elastic {i + 1}")
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.plot(x_countmin, np.gradient(y_countmin, packets_per_log), label=f"elastic {i + 1}")

    result_dynamic = execute_command([
        "--type", "dynamic",
        "--width", "28",
        "--depth", "300",
        "--repeat", "modify_size_cyclic", str(packets_per_modify_size), str(count_sketch),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_dynamic = np.array(result_dynamic['index'].to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_dynamic_countmin")
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.plot(x_dynamic, np.gradient(y_dynamic, packets_per_log), label=f"dynamic")


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
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.savefig('mae_dynamic_countmin.png')

    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.legend()
    plt.grid()
    plt.ylabel("MAE'")
    plt.xlabel('packets')
    plt.savefig('mae_dynamic_countmin_derivative.png')

    plt.show()


def plot_mae_elastic_shrink(count_sketch: int, count_log: int):
    packets_per_shrink = COUNT_PACKETS // count_sketch
    packets_per_log = COUNT_PACKETS // count_log

    for i in range(count_sketch):
        factor = pow(2,i)
        result_countmin = execute_command([
            "--type", "elastic",
            "--width", str(28 * factor),
            "--depth", "300",
            "--repeat", "log_mean_absolute_error", str(packets_per_log),
        ])
        x_countmin = np.array(result_countmin['index'].to_numpy())
        y_countmin = np.array(result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_elastic_shrink")
        plt.plot(x_countmin, y_countmin, label=f"elastic {factor}")
        plt.figure(num="mae_elastic_shrink_derivative")
        plt.plot(x_countmin, np.gradient(y_countmin, packets_per_log), label=f"elastic {factor}")

    result_elastic_shrink = execute_command([
        "--type", "elastic",
        "--width", str(28*pow(2, count_sketch-1)),
        "--depth", "300",
        "--repeat", "shrink", str(packets_per_shrink),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_elastic_shrink = np.array(result_elastic_shrink['index'].to_numpy())
    y_elastic_shrink = np.array(result_elastic_shrink['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_elastic_shrink")
    plt.plot(x_elastic_shrink, y_elastic_shrink, label="elastic shrink")
    plt.figure(num="mae_elastic_shrink_derivative")
    plt.plot(x_elastic_shrink, np.gradient(y_elastic_shrink, packets_per_log), label=f"elastic*")


    for xc in range(packets_per_shrink, packets_per_shrink * count_sketch, packets_per_shrink):
        plt.figure(num="mae_elastic_shrink")
        plt.axvline(x=xc, color='r', linestyle='dashed')
        plt.figure(num="mae_elastic_shrink_derivative")
        plt.axvline(x=xc, color='r', linestyle='dashed')

    plt.figure(num="mae_elastic_shrink")
    plt.legend()
    plt.grid()
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.title('mae_elastic_shrink')
    plt.savefig('mae_elastic_shrink.png')

    plt.figure(num="mae_elastic_shrink_derivative")
    plt.legend()
    plt.grid()
    plt.ylabel("MAE'")
    plt.xlabel('packets')
    plt.savefig('mae_elastic_shrink_derivative.png')
    plt.title('mae_elastic_shrink_derivative')

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

def plot_dynamic_sketches_loads(dynamic_max_size: int, count_buckets):
    packets_per_expand = COUNT_PACKETS // dynamic_max_size
    result = execute_command([
        "--type", "dynamic",
        "--buckets", str(count_buckets),
        "--repeat", "log_dynamic_sketches_loads", str(packets_per_expand),
        "--repeat", "expand", str(packets_per_expand)
    ])
    last_state = result.iloc[-1]
    loads = last_state['loads']
    loads.sort(key=lambda l: (l['min_key'], l['min_key']-l['max_key']))
    num_events_max = max([n['num_events'] for n in loads])
    updates_since_last_clear_max = max([n['updates_since_last_clear'] for n in loads])
    for i in range(len(loads)):
        num_events = loads[i]['num_events']
        updates_since_last_clear = loads[i]['updates_since_last_clear']
        min_key = loads[i]['min_key']
        max_key = loads[i]['max_key']
        color = num_events / num_events_max
        print(i, min_key, max_key, max_key-min_key, num_events)
        plt.hlines(y=i, xmin=min_key, xmax=max_key,linewidth=4, color=(color,color,0))
    plt.title('sketches ranges and load')
    plt.xlabel('ip as integer')
    plt.ylabel('sketch index')
    plt.show()

def plot_dynamic_sketches_count(dynamic_max_size: int):
    packets_per_expand = COUNT_PACKETS // dynamic_max_size
    result = execute_command([
        "--type", "dynamic",
        "--repeat", "log_dynamic_sketches_loads", str(packets_per_expand),
        "--repeat", "expand", str(packets_per_expand)
    ])
    last_state = result.iloc[-1]
    loads = last_state['loads']
    loads.sort(key=lambda l: (l['min_key'], l['min_key']-l['max_key']))
    xs = list(set([l['min_key'] for l in loads]).union(set([l['max_key'] for l in loads])))
    xs.sort()
    ys = [0 for x in xs]
    for l in loads:
        l_min = l['min_key']
        l_max = l['max_key']
        i_min = xs.index(l_min)
        i_max = xs.index(l_max)
        print(i_min, i_max)
        for i in range(i_min, i_max + 1):
            ys[i] += 1
    plt.title("sketches count")
    plt.plot(xs, ys)
    plt.xlabel('ip as integer')
    plt.ylabel('sketch count')
    plt.show()

def plot_mae_dynamic_and_elastic(count_log, count_expand, count_buckets):
    packets_per_log = COUNT_PACKETS // count_log
    packets_per_expand = COUNT_PACKETS // count_expand

    result_dynamic = execute_command([
        "--type", "dynamic",
        "--buckets", str(count_buckets),
        "--repeat", "expand", str(packets_per_expand),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    result_elastic = execute_command([
        "--type", "elastic",
        "--buckets", str(count_buckets),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_dynamic = np.array(result_dynamic['index'].to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())

    x_elastic = np.array(result_elastic['index'].to_numpy())
    y_elastic = np.array(result_elastic['log_mean_absolute_error'].to_numpy())

    plt.grid()
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.plot(x_elastic, y_elastic, label="elastic")
    plt.legend()
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.savefig('mae_dynamic_and_elastic.png')

    plt.show()

if __name__ == "__main__":
    #plot_mae_dynamic_and_elastic(128, 8, 8)
    #plot_mae_elastic_shrink(3, 128)
    plot_mae_dynamic_and_countmin(3, 128)
    #plot_ip_distribution()
    #plot_dynamic_sketches_loads(16, 8)
    #plot_dynamic_sketches_count(16)
    #plot_operations_per_second(64)
