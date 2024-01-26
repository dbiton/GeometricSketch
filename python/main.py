import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import subprocess as sp
import json
import pandas as pd
from collections import Counter
import os
from numpy import log10
from scipy.optimize import curve_fit
from scipy.special import zetac

if os.name == 'nt':
    filepath_packets = '..\\pcaps\\capture.txt'
    filepath_executable = "..\\cpp\\x64\\Release\\DynamicSketch.exe"
else:
    filepath_packets = '../pcaps/capture.txt'
    filepath_executable = "../cpp/build-DynamicSketch-Desktop-Debug/DynamicSketch"

COUNT_PACKETS_MAX = 33000000
COUNT_PACKETS = min(5000, COUNT_PACKETS_MAX)


def execute_command(arguments: list):
    command = [filepath_executable, "--limit_file",
               filepath_packets, str(COUNT_PACKETS)] + arguments
    print("executing:", ' '.join(command))
    result = sp.run(command, stdout=sp.PIPE,
                    stderr=sp.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise ValueError(
            f"command: {' '.join(command)} caused error: {result.stderr}")
    raw_output = result.stdout.replace("\n", "").replace("\t", "")[:-2] + ']'
    output_dict = json.loads(raw_output)
    output_df = pd.DataFrame(output_dict).groupby('index').mean()
    return output_df


def get_packets(filepath_packets: str):
    file_packets = open(filepath_packets, "r")
    return np.array([int(ip) for ip in file_packets.readlines()])


def zipfian(x):
    return 10 ** (-x+5.5)

def plot_update_query_throughput(B: int, L: int):
    sketch_width = 500
    sketch_depth = 5

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    count_log_time = 16
    count_query = 64
    packets_per_log_time = COUNT_PACKETS // count_log_time
    packets_per_query = COUNT_PACKETS // count_query
    throughputs_update = np.zeros((L, B-2))
    throughputs_query = np.zeros((L, B-2))
    for l in range(L):
        for b in range(2, B):
            expand_size = sketch_width * ((b**(l+1)-1)/(b-1)) - sketch_width
            result = execute_command([
                "--type", "cellsketch",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--repeat", "log_update_time", str(packets_per_log_time),
                "--repeat", "log_query_time", str(packets_per_log_time),
                "--repeat", "log_average_relative_error", str(packets_per_query)])

            time_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            time_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughputs_update[l, b-2] = round(1 / time_per_update)
            throughputs_query[l, b-2] = round(1 / time_per_query)

    ax0.imshow(throughputs_update, origin='lower', extent=[2,B,0,L])
    ax1.imshow(throughputs_query, origin='lower', extent=[2,B,0,L])

    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i+2.5, j+0.5, int(label), ha='center', va='center')

    ax0.set_title('update')
    ax0.set_xlabel('Branching Factor')
    ax0.set_ylabel('Layers')
    ax1.set_title('query')
    ax1.set_xlabel('Branching Factor')
    ax1.set_ylabel('Layers')
    plt.show()

def plot_ip_distribution():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    
    packets = get_packets(filepath_packets)
    uint32_max = 0xffffffff
    bin_count = 64
    bins = np.arange(0, uint32_max, uint32_max / bin_count)
    counts, bins = np.histogram(packets, bins=bins)
    ax1.set_ylabel('frequency')
    ax1.set_xlabel('IP')
    ax1.stairs(counts, bins)

    counter = Counter(packets)
    frequency = np.array(sorted(list(counter.values()), key=lambda x: -x))
    rank = np.arange(1, len(frequency)+1)
    ax0.plot(rank, frequency)
    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_xlabel('frequency rank')
    ax0.set_ylabel('frequency')
    #ax0.plot(rank, zipfian(rank), label='10^(5.5-r)')

    plt.show()


def plot_mae_cellsketch_expand(count_sketch: int, count_log: int):
    packets_per_modify_size = COUNT_PACKETS // count_sketch
    packets_per_log = COUNT_PACKETS // count_log
    width = 1000
    depth = 5

    result_dynamic = execute_command([
        "--type", "cellsketch",
        "--width", str(width),
        "--depth", str(depth),
        "--repeat", "expand", str(packets_per_modify_size),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    result_count_min = execute_command([
        "--type", "countmin",
        "--repeat", "log_mean_absolute_error", str(packets_per_log),
        "--width", str(width),
        "--depth", str(depth)])

    result_count_max = execute_command([
        "--type", "countmin",
        "--repeat", "log_mean_absolute_error", str(packets_per_log),
        "--width", str(width*count_sketch),
        "--depth", str(depth)])

    x_dynamic = np.array(result_dynamic.index.to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())
    x_count_min = np.array(result_count_min.index.to_numpy())
    y_count_min = np.array(
        result_count_min['log_mean_absolute_error'].to_numpy())
    x_count_max = np.array(result_count_max.index.to_numpy())
    y_count_max = np.array(result_count_max['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_dynamic_countmin")
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.plot(x_count_min, y_count_min, label="min")
    plt.plot(x_count_max, y_count_max, label="max")
    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.plot(x_dynamic, np.gradient(
        y_dynamic, packets_per_log), label=f"dynamic")
    plt.plot(x_count_min, np.gradient(
        y_count_min, packets_per_log), label=f"min")
    plt.plot(x_count_max, np.gradient(
        y_count_max, packets_per_log), label=f"max")

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


def plot_mae_dynamic_and_linked_cell(count_sketch: int, count_log: int):
    packets_per_modify_size = COUNT_PACKETS // count_sketch
    packets_per_log = COUNT_PACKETS // count_log

    for i in range(count_sketch):
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(250 * (i + 1)),
            "--depth", "4",
            "--repeat", "log_mean_absolute_error", str(packets_per_log),
        ])
        x_countmin = np.array(result_countmin.index.to_numpy())
        y_countmin = np.array(
            result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_dynamic_countmin")
        plt.plot(x_countmin, y_countmin, label=f"countmin {i + 1}")
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.plot(x_countmin, np.gradient(
            y_countmin, packets_per_log), label=f"countmin {i + 1}")

    result_dynamic = execute_command([
        "--type", "cellsketch",
        "--width", "250",
        "--depth", "4",
        "--repeat", "expand", str(packets_per_modify_size),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_dynamic = np.array(result_dynamic.index.to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_dynamic_countmin")
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.plot(x_dynamic, np.gradient(
        y_dynamic, packets_per_log), label=f"dynamic")

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
        x_countmin = np.array(result_countmin.index.to_numpy())
        y_countmin = np.array(
            result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_dynamic_countmin")
        plt.plot(x_countmin, y_countmin, label=f"elastic {i + 1}")
        plt.figure(num="mae_dynamic_countmin_derivative")
        plt.plot(x_countmin, np.gradient(
            y_countmin, packets_per_log), label=f"elastic {i + 1}")

    result_dynamic = execute_command([
        "--type", "dynamic",
        "--width", "28",
        "--depth", "300",
        "--repeat", "modify_size_cyclic", str(
            packets_per_modify_size), str(count_sketch),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_dynamic = np.array(result_dynamic.index.to_numpy())
    y_dynamic = np.array(result_dynamic['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_dynamic_countmin")
    plt.plot(x_dynamic, y_dynamic, label="dynamic")
    plt.figure(num="mae_dynamic_countmin_derivative")
    plt.plot(x_dynamic, np.gradient(
        y_dynamic, packets_per_log), label=f"dynamic")

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
        factor = pow(2, i)
        result_countmin = execute_command([
            "--type", "elastic",
            "--width", str(28 * factor),
            "--depth", "300",
            "--repeat", "log_mean_absolute_error", str(packets_per_log),
        ])
        x_countmin = np.array(result_countmin.index.to_numpy())
        y_countmin = np.array(
            result_countmin['log_mean_absolute_error'].to_numpy())
        plt.figure(num="mae_elastic_shrink")
        plt.plot(x_countmin, y_countmin, label=f"elastic {factor}")
        plt.figure(num="mae_elastic_shrink_derivative")
        plt.plot(x_countmin, np.gradient(
            y_countmin, packets_per_log), label=f"elastic {factor}")

    result_elastic_shrink = execute_command([
        "--type", "elastic",
        "--width", str(28*pow(2, count_sketch-1)),
        "--depth", "300",
        "--repeat", "shrink", str(packets_per_shrink),
        "--repeat", "log_mean_absolute_error", str(packets_per_log)])

    x_elastic_shrink = np.array(result_elastic_shrink.index.to_numpy())
    y_elastic_shrink = np.array(
        result_elastic_shrink['log_mean_absolute_error'].to_numpy())
    plt.figure(num="mae_elastic_shrink")
    plt.plot(x_elastic_shrink, y_elastic_shrink, label="elastic shrink")
    plt.figure(num="mae_elastic_shrink_derivative")
    plt.plot(x_elastic_shrink, np.gradient(
        y_elastic_shrink, packets_per_log), label=f"elastic*")

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


def plot_mae_countmin_and_linked_cell():
    result_countmin = execute_command(
        ["--type", "countmin", "--repeat", "log_mean_absolute_error", "1000", "--repeat", "log_memory_usage", "1000"])
    result_linkedcell = execute_command(
        ["--type", "cellsketch", "--repeat", "log_mean_absolute_error", "1000", "--repeat", "log_memory_usage", "1000"])
    x_countmin = np.array(result_countmin.index.to_numpy())
    y_countmin = np.array(
        result_countmin['log_mean_absolute_error'].to_numpy())
    x_countsketch = np.array(result_linkedcell.index.to_numpy())
    y_countsketch = np.array(
        result_linkedcell['log_mean_absolute_error'].to_numpy())
    plt.plot(x_countmin, y_countmin, label="countmin")
    plt.plot(x_countsketch, y_countsketch, label="cellsketch")
    plt.ylabel('MAE')
    plt.xlabel('packets')
    plt.title('packets distribution')
    plt.legend()
    plt.grid()
    plt.show()


def plot_branching_factor(branching_factors: list, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = max(branching_factors) * sketch_width
    packets_per_expand = COUNT_PACKETS // counters_added_per_row
    result_countmin = execute_command([
        "--type", "countmin",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--repeat", "log_memory_usage", str(packets_per_log)])
    y_mae = np.array(result_countmin['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_countmin['memory_usage'].to_numpy())
    x = np.array(result_countmin.index.to_numpy())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax0.plot(x, y_mem, label="CMS", marker='o')
    ax1.plot(x, y_mae, label="CMS", marker='o')
    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')

    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    markers = ["D", "s", "^"]
    for i, branching_factor in enumerate(branching_factors):
        result_cellsketch = execute_command([
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(branching_factor),
            "--repeat", "expand", str(packets_per_expand), "0",
            "--repeat", "compress", str("1000000"), str(branching_factor),
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--repeat", "log_memory_usage", str(packets_per_log)])
        y_mae = np.array(
            result_cellsketch['log_average_relative_error'].to_numpy())
        y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
        x = np.array(result_cellsketch.index.to_numpy())
        ax0.plot(x, y_mem, label=f'GS-{branching_factor}', marker=markers[i])
        ax1.plot(x, y_mae, label=f'GS-{branching_factor}', marker=markers[i])
    ax0.legend()
    ax1.legend()
    plt.show()


def plot_gs_cms_comparison(B: int, L: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = sketch_width * (B**L-1) / (B-1)
    packets_per_expand = floor(COUNT_PACKETS / counters_added_per_row)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["D", "s", "^"]
    for l in range(L):
        cm_width = sketch_width * B**l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--repeat", "log_memory_usage", str(packets_per_log)])
        y_mae = np.array(
            result_countmin['log_average_relative_error'].to_numpy())
        y_mem = np.array(result_countmin['memory_usage'].to_numpy())
        x = np.array(result_countmin.index.to_numpy())
        ax0.plot(x, y_mem, label=f"CMS-{l}", marker=markers[l])
        ax1.plot(x, y_mae, label=f"CMS-{l}", marker=markers[l])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    result_cellsketch = execute_command([
        "--type", "cellsketch",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--repeat", "expand", str(packets_per_expand), "0",
        "--repeat", "compress", str(packets_per_expand), str(1000000),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--repeat", "log_memory_usage", str(packets_per_log)])
    y_mae = np.array(
        result_cellsketch['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
    x = np.array(result_cellsketch.index.to_numpy())
    ax0.plot(x, y_mem, label=f'GS', marker='o')
    ax1.plot(x, y_mae, label=f'GS', marker='o')
    ax0.legend()
    ax1.legend()
    plt.show()


def plot_gs_cms_derivative_comparison(B: int, L: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = sketch_width * (B**L-1) / (B-1) - sketch_width
    packets_per_expand = floor(COUNT_PACKETS / counters_added_per_row)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    markers = ["D", "s", "^", "o", "p"]
    for l in range(L):
        cm_width = sketch_width * B**l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ])
        y_mae = result_countmin['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_countmin['memory_usage'].dropna().to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        x = result_countmin.index.to_numpy()
        ax0.plot(x, y_mem, label=f"CMS-{l}", marker=markers[l])
        ax1.plot(x, y_mae, label=f"CMS-{l}", marker=markers[l])
        ax2.plot(x, d_mae, label=f"CMS-{l}", marker=markers[l])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax2.grid()
    ax2.set_ylabel("ARE'")
    ax2.set_xlabel('Packets Count')
    command = [
        "--type", "cellsketch",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--repeat", "expand", str(packets_per_expand), "0",
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_memory_usage", "0",
        "--once", "log_average_relative_error", "0",
        "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
        "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
        "--repeat", "log_memory_usage", str(packets_per_log),
        "--repeat", "compress", str(packets_per_expand), str(1000000)
    ]
    result_cellsketch = execute_command(command)
    y_mae = result_cellsketch['log_average_relative_error'].dropna().to_numpy()
    y_mem = result_cellsketch['memory_usage'].dropna().to_numpy()
    d_mae = np.gradient(y_mae, packets_per_log)
    x = result_cellsketch.index.to_numpy()
    ax0.plot(x, y_mem, label=f'GS-{B}', marker="o")
    ax1.plot(x, y_mae, label=f'GS-{B}', marker="o")
    ax2.plot(x, d_mae, label=f"GS-{B}", marker="o")
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_gs_cms_static_comparison(B: int, L: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["s", "p", "d", "D", "v", "^", ">", "<"]
    for l in range(1, L):
        cm_width = sketch_width * B**l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ])
        y_mae = result_countmin['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_countmin['memory_usage'].dropna().to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        x = result_countmin.index.to_numpy()
        ax0.plot(x, y_mem, label=f"CMS-{l}", marker=markers[l])
        ax1.plot(x, y_mae, label=f"CMS-{l}", marker=markers[l])

        counters_added_per_row = int(sketch_width * (B**(l+1)-1) / (B-1) - sketch_width)
        command = [
        "--type", "cellsketch",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--once", "expand", "0", str(counters_added_per_row),
        "--once", "compress", "0", "100000000",
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_memory_usage", "0",
        "--once", "log_average_relative_error", "0",
        "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
        "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
        "--repeat", "log_memory_usage", str(packets_per_log),
        ]
        result_cellsketch = execute_command(command)
        y_mae = result_cellsketch['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_cellsketch['memory_usage'].dropna().to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        x = result_cellsketch.index.to_numpy()
        ax0.plot(x, y_mem, label=f'GS-{l}', marker=markers[L+l])
        ax1.plot(x, y_mae, label=f'GS-{l}', marker=markers[L+l])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_gs_dcms_comparison(B: int, K:int, N: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["v", "^", ">", "<"]
    strategies = [
        lambda j: K*j,
        lambda j: K**j,
    ]
    for i, strategy in enumerate(strategies):
        command_dcms = [
            "--type", "dynamic",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ]
        command_gs = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ]
        j = 1
        total = 0
        while True:
            v = strategy(j)
            thresh = N*v
            total += thresh
            appended_sketch_width = sketch_width * v
            appended_counters_count = appended_sketch_width #* B / (B-1)
            if total >= COUNT_PACKETS:
                break
            compress_command = [] #["--once", "compress", str(total), "1000000000"]
            command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
            command_gs += ["--once", "expand", str(total), str(appended_counters_count)] + compress_command
            j+=1

        for args in [(0, "DCMS", command_dcms), (1, "GS", command_gs)]:
            s, type, command = args 
            result = execute_command(command)
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            x = result.index.to_numpy()
            ax0.plot(x, y_mem, label=f"{type}-{i}", marker=markers[i+s*2])
            ax1.plot(x, y_mae, label=f"{type}-{i}", marker=markers[i+s*2])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_gs_dcms_granular_comparison(B: int, K:int, N: int, granularity: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["s", "p", "d", "D", "v", "^", ">", "<"]
    strategies = [
        lambda j: K*j,
        lambda j: K**j,
    ]
    for i, strategy in enumerate(strategies):
        command_dcms = [
            "--type", "dynamic",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ]
        command_gs = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1)
        ]
        j = 1
        total = 0
        while True:
            v = strategy(j)
            prev_total = total
            total += N*v
            appended_sketch_width = sketch_width * v
            total_appended_counters_count = appended_sketch_width * B / (B-1)
            if total >= COUNT_PACKETS:
                break
            command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
            for g in range(1,granularity):
                index = prev_total + g/granularity*(total-prev_total)
                appended_counters_count = total_appended_counters_count / granularity
                compress_command = ["--once", "compress", str(index), "1000000000"]
                command_gs += ["--once", "expand", str(index), str(appended_counters_count)] + compress_command
            j+=1

        for args in [(0, "DCMS", command_dcms), (1, "GS", command_gs)]:
            s, type, command = args 
            result = execute_command(command)
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            d_mae = np.gradient(y_mae, packets_per_log)
            x = result.index.to_numpy()
            ax0.plot(x, y_mem, label=f"{type}-{i}", marker=markers[i+s*2])
            ax1.plot(x, y_mae, label=f"{type}-{i}", marker=markers[i+s*2])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_gs_undo_expand(B: int, L: int, max_cycles: int, granularity: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["s", "p", "d", "D", "v", "^", ">", "<"]
    for cycle_counts in range(1, max_cycles+1):
        counters_added_per_row = int(sketch_width * (B**L-1) / (B-1) - sketch_width)
        command = [
        "--type", "cellsketch",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_memory_usage", "0",
        "--once", "log_average_relative_error", "0",
        "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
        "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
        "--repeat", "log_memory_usage", str(packets_per_log)
        ]
        
        for cycle in range(cycle_counts):
            counters_chunk = round(counters_added_per_row / granularity)
            packet_chunk = round((COUNT_PACKETS - 2) / cycle_counts)
            cycle_begin = int(packet_chunk * cycle)
            cycle_end = int(packet_chunk * (cycle+1))
            cycle_mid = int((cycle_begin + cycle_end) / 2)
            commands =  [["--once", "expand", str(cycle_mid - int(packet_chunk / 2 * i / granularity)), str(counters_chunk)] for i in range(granularity)]
            commands += [["--once", "shrink", str(cycle_end - int(packet_chunk / 2 * i / granularity)), str(counters_chunk)] for i in range(granularity)]
            command += sum(commands, [])

        result_cellsketch = execute_command(command)
        y_mae = result_cellsketch['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_cellsketch['memory_usage'].dropna().to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        x = result_cellsketch.index.to_numpy()
        ax0.plot(x, y_mem, label=f'GS-B{B}-C{cycle_counts}', marker=markers[cycle_counts])
        ax1.plot(x, y_mae, label=f'GS-B{B}-C{cycle_counts}', marker=markers[cycle_counts])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax1.legend()
    fig.tight_layout()
    plt.show()

def plot_gs_dynamic_compression_comparison(B: int, L: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    packets_per_modify = math.floor(COUNT_PACKETS / L / 2) 

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    expand_sizes = [sketch_width * B ** l for l in range(1,L)]
    command_lines_expand   = sum([["--once", "expand", str((i+1)*packets_per_modify), str(e)] for i, e in enumerate(expand_sizes)], [])
    command_lines_shrink   = sum([["--once", "shrink", str((i+1)*packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in enumerate(reversed(expand_sizes))], [])
    command_lines_compress = sum([["--once", "compress", str((i+1)*packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in enumerate(reversed(expand_sizes))], [])
    command_countmin = [
        "--type", "dynamic",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--repeat", "log_memory_usage", str(packets_per_log)
    ] + command_lines_expand
    command_geometric = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--repeat", "log_memory_usage", str(packets_per_log)
    ] + command_lines_expand 


    markers = ['D', 'X', 's', '^']
    names = ["shrink", "compress"]
    names_type = ["GS", "DCMS"]
    for j, command_lines0 in enumerate([command_geometric, command_countmin]):
        for i, command_lines1 in enumerate([command_lines_shrink, command_lines_compress]):
            result = execute_command(
                command_lines0 + command_lines1
            )
            y_mae = np.array(
                result['log_average_relative_error'].to_numpy())
            y_mem = np.array(result['memory_usage'].to_numpy())
            x = np.array(result.index.to_numpy())
            ax0.plot(x, y_mem, label=f'{names_type[j]}-{names[i]}', marker=markers[i+j*2])
            ax1.plot(x, y_mae, label=f'{names_type[j]}-{names[i]}', marker=markers[i+j*2])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax0.legend()
    ax1.legend()
    plt.show()

# replace with undo expand
def plot_gs_dynamic_expand_comparison(B: int, L: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    packets_per_modify = floor(COUNT_PACKETS / L)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    expand_sizes = [sketch_width * B ** l for l in range(1,L)]
    command_lines_expand   = sum([["--once", "expand", str((i+1)*packets_per_modify), str(e)] for i, e in enumerate(expand_sizes)], [])
    command_lines_compress = sum([["--once", "compress", str((i+1)*packets_per_modify), str(e)] for i, e in enumerate(reversed(expand_sizes))], [])
    command_countmin = [
        "--type", "dynamic",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--repeat", "log_memory_usage", str(packets_per_log)
    ] + command_lines_expand + command_lines_compress

    command_geometric = [
        "--type", "cellsketch",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--repeat", "log_memory_usage", str(packets_per_log)
    ] + command_lines_expand + command_lines_compress
    result_cellsketch = execute_command(
        command_geometric
    )
    y_mae = np.array(
        result_cellsketch['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
    x = np.array(result_cellsketch.index.to_numpy())
    ax0.plot(x, y_mem, label=f'GS', marker='^')
    ax1.plot(x, y_mae, label=f'GS', marker='^')

    result_countmin = execute_command(command_countmin)
    y_mae = np.array(
        result_countmin['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_countmin['memory_usage'].to_numpy())
    x = np.array(result_countmin.index.to_numpy())
    ax0.plot(x, y_mem, label=f"DCMS", marker='D')
    ax1.plot(x, y_mae, label=f"DCMS", marker='D')

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax0.legend()
    ax1.legend()
    plt.show()

def plot_gs_derivative(B: int, L: int, count_expand: int, count_log: int):
    sketch_width = 500
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    packets_per_expand = COUNT_PACKETS // count_expand

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel("ARE")
    ax1.set_xlabel('Packets Count')
    ax2.grid()
    ax2.set_ylabel("ARE'")
    ax2.set_xlabel('Packets Count')
    markers = ["D", "s", "^", "*", "X"]

    N = sketch_width * (B**L-1) / (B-1)
    expand_functions = [
        lambda x: N * math.sqrt(x),
        lambda x: N * math.log2(x+1),
        lambda x: N * (2**x-1),
        lambda x: N * math.sin(math.sqrt(x)*math.pi/2),
        lambda x: N * math.asin(x*x) * 2 / math.pi
    ]

    for i, expand_function in enumerate(expand_functions):
        command = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS-1),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS-1)]
        expands = []
        for expand_index in range(1, count_expand):
            packet_index = expand_index*packets_per_expand
            expand_size = expand_function(expand_index/count_expand) - sum(expands)
            expands.append(expand_size)
            command += ["--once", "expand", str(packet_index), str(expand_size)]
            command += ["--once", "compress", str(packet_index), str(1000000)]
        result_cellsketch = execute_command(command)
        y_mae = np.array(
            result_cellsketch['log_average_relative_error'].to_numpy())
        y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
        d_mae = np.gradient(y_mae, packets_per_log)
        x = np.array(result_cellsketch.index.to_numpy())
        ax0.plot(x, y_mem, label=f"f-{i}", marker=markers[i])
        ax1.plot(x, y_mae, label=f"f-{i}", marker=markers[i])
        ax2.plot(x, d_mae, label=f"f-{i}", marker=markers[i])
    ax1.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_predict_distribution()
    # plot_mae_dynamic_and_elastic(128, 8, 8)
    # plot_mae_elastic_shrink(3, 128)
    # plot_mae_dynamic_and_countmin(3, 128)
    #plot_ip_distribution()
    # plot_ip_distribution_zipfian()
    # plot_dynamic_sketches_loads(16, 8)
    # plot_dynamic_sketches_count(16)
    # plot_operations_per_second(64)
    # plot_mae_countmin_and_countsketch()
    # plot_mae_countmin_and_linked_cell()
    # plot_mae_dynamic_and_linked_cell(4, 128)
    # plot_mae_cellsketch_expand(2, 128)

    # plot_branching_factor([2,4,8], 16)
    # plot_gs_cms_comparison(2, 3, 16)
    # plot_gs_dynamic_compression_comparison(2, 5, 16)
    # plot_update_query_throughput(9, 7)
    # plot_gs_cms_derivative_comparison(2, 4, 16)
    # plot_gs_derivative(2,3,512,16)
    # plot_gs_undo_expand(B=2, L=3, max_cycles=3, granularity=256, count_log=64)
    # plot_gs_cms_static_comparison(2, 4, 16)
    plot_gs_dcms_comparison(2, 2, 640, 4)
    # plot_gs_dcms_granular_comparison(2,5,500,64,32)