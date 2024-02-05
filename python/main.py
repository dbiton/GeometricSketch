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
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

if os.name == 'nt':
    filepath_zipf = '..\\pcaps\\zipf'
    filepath_packets = '..\\pcaps\\capture.txt'
    filepath_executable = "..\\cpp\\x64\\Release\\DynamicSketch.exe"
else:
    filepath_zipf = '../pcaps/zipf'
    filepath_packets = '../pcaps/capture.txt'
    filepath_executable = "../cpp/build-DynamicSketch-Desktop-Release/DynamicSketch"

COUNT_PACKETS_MAX = 37700000
COUNT_PACKETS = min(37700000, COUNT_PACKETS_MAX)


def execute_command(arguments: list, packets_path=filepath_packets):
    command = [filepath_executable, "--limit_file",
               packets_path, str(COUNT_PACKETS)] + arguments
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


def plot_cms_update_query_throughput(min_width_pow: int, count_width: int, count_depth: int, figure_name: str):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    packets_per_log_time = COUNT_PACKETS - 1
    throughputs_update = np.zeros((count_width, count_depth))
    throughputs_query = np.zeros((count_width, count_depth))
    for D in range(1, count_depth + 1):
        for sketch_width_pow in range(min_width_pow, min_width_pow + count_width):
            W = 2 ** sketch_width_pow
            result = execute_command([
                "--type", "countmin",
                "--width", str(W),
                "--depth", str(D),
                "--repeat", "log_update_time", str(packets_per_log_time),
                "--repeat", "log_query_time", str(packets_per_log_time)])
            ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            throughput_update = 1 / (ms_per_update * 10e3)
            ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughput_query = 1 / (ms_per_query * 10e3)
            throughputs_update[sketch_width_pow - min_width_pow, D - 1] = throughput_update
            throughputs_query[sketch_width_pow - min_width_pow, D - 1] = throughput_query
            print(D, sketch_width_pow)

    im0 = ax0.imshow(throughputs_update, origin='lower',
                     extent=[1, count_depth + 1, min_width_pow, min_width_pow + count_width])
    im1 = ax1.imshow(throughputs_query, origin='lower',
                     extent=[1, count_depth + 1, min_width_pow, min_width_pow + count_width])

    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 0.5 + 1, j + 0.5 + min_width_pow, int(label), ha='center', va='center')

    ax0.set_title('CMS Update MOPS')
    ax0.set_xlabel('Depth')
    ax0.set_ylabel('Log2(Width)')
    ax1.set_title('CMS Query MOPS')
    ax1.set_xlabel('Depth ')
    ax1.set_ylabel('Log2(Width)')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_update_query_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    throughputs_update = np.zeros((L, B - 2))
    throughputs_query = np.zeros((L, B - 2))
    for l in range(L):
        for b in range(2, B):
            expand_size = sketch_width * ((b ** (l + 1) - 1) / (b - 1)) - sketch_width
            result = execute_command([
                "--type", "cellsketch",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--once", "log_update_time", str(COUNT_PACKETS - 1),
                "--once", "log_query_time", str(COUNT_PACKETS - 1)])

            ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            throughputs_update[l, b - 2] = 10 / (ms_per_update * 10e3)
            ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughputs_query[l, b - 2] = 10 / (ms_per_query * 10e3)
    im0 = ax0.imshow(throughputs_update, origin='lower', extent=[2, B, 0, L])
    im1 = ax1.imshow(throughputs_query, origin='lower', extent=[2, B, 0, L])

    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 0.5, int(label), ha='center', va='center')

    ax0.set_title('GS Update 10*MOPS')
    ax0.set_xlabel('Branching Factor')
    ax0.set_ylabel('Layers')
    ax1.set_title('GS Query 10*MOPS')
    ax1.set_xlabel('Branching Factor')
    ax1.set_ylabel('Layers')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_expand_undo_compress_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax_expand, ax_shrink, ax_compress) = plt.subplots(1, 3, figsize=(12, 4))
    count_log_time = 16
    count_query = 64
    packets_per_log_time = COUNT_PACKETS // count_log_time
    packets_per_query = COUNT_PACKETS // count_query
    throughputs_compress = np.zeros((L, B - 2))
    throughputs_expand = np.zeros((L, B - 2))
    throughputs_shrink = np.zeros((L, B - 2))
    for l in range(L):
        for b in range(2, B):
            expand_size = sketch_width * ((b ** (l + 1) - 1) / (b - 1)) - sketch_width
            next_layer_width = sketch_width * b ** (l + 1)
            result = execute_command([
                "--type", "cellsketch",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--once", "log_expand_and_shrink_time", "0", str(next_layer_width)
            ])
            ms_per_expand = np.array(result['log_expand_time'].dropna().to_numpy()).mean()
            throughputs_expand[l, b - 2] = log10(1 / (ms_per_expand * 10e3))
            ms_per_shrink = np.array(result['log_shrink_time'].dropna().to_numpy()).mean()
            throughputs_shrink[l, b - 2] = log10(1 / (ms_per_shrink * 10e3))
    for l in range(1, L + 1):
        for b in range(2, B):
            expand_size = sketch_width * ((b ** (l + 1) - 1) / (b - 1)) - sketch_width
            curr_layer_width = sketch_width * b ** l
            result = execute_command([
                "--type", "cellsketch",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--once", "log_compress_time", "0", str(curr_layer_width)
            ])
            ms_per_compress = np.array(result['log_compress_time'].dropna().to_numpy()).mean()
            throughputs_compress[l - 1, b - 2] = log10(1 / (ms_per_compress * 10e3))
    im0 = ax_expand.imshow(throughputs_expand, origin='lower', extent=[2, B, 0, L])
    im1 = ax_shrink.imshow(throughputs_shrink, origin='lower', extent=[2, B, 0, L])
    im2 = ax_compress.imshow(throughputs_compress, origin='lower', extent=[2, B, 1, L + 1])

    for (throughputs, ax) in [(throughputs_expand, ax_expand), (throughputs_shrink, ax_shrink)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 0.5, int(label), ha='center', va='center')

    for (j, i), label in np.ndenumerate(throughputs_compress):
        ax_compress.text(i + 2.5, j + 1.5, int(label), ha='center', va='center')

    ax_expand.set_title('GS Expand Log10(MOPS)')
    ax_expand.set_xlabel('Branching Factor')
    ax_expand.set_ylabel('Layers')
    ax_shrink.set_title('GS Shrink Log10(MOPS)')
    ax_shrink.set_xlabel('Branching Factor')
    ax_shrink.set_ylabel('Layers')
    ax_compress.set_title('GS Compress Log10(MOPS)')
    ax_compress.set_xlabel('Branching Factor')
    ax_compress.set_ylabel('Layers')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_ip_distribution_zipf(figure_name: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    step = 0.01
    max_a = 2.29
    cmap = cm.get_cmap("Spectral")
    norm = colors.Normalize(1, max_a)
    for a_float in np.arange(1 + step, max_a, step):
        a = np.round(a_float, 2)
        print("ip", a)
        packets = get_packets(f"../pcaps/zipf/zipf-{a}.txt")
        counter = Counter(packets)
        frequency = np.array(sorted(list(counter.values()), key=lambda x: -x)) / len(packets)
        rank = np.arange(1, len(frequency) + 1)
        ax.plot(rank, frequency, color=cmap(norm(a)))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Zipf Parameter")
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def zipf(x, a):
    return np.float_power(x, -a) / zetac(a)


def plot_ip_distribution(figure_name: str):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    packets = get_packets(filepath_packets)
    uint32_max = 0xffffffff
    bin_count = 128
    bins = np.arange(0, uint32_max, uint32_max / bin_count)
    counts, bins = np.histogram(packets, bins=bins)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('IP')
    ax1.stairs(counts, bins, fill=True)

    counter = Counter(packets)
    packets_counts = np.array(sorted(list(counter.values()), key=lambda x: -x))
    frequency = packets_counts / len(packets)
    rank = np.arange(1, len(frequency) + 1)
    ax0.plot(rank, frequency, label="Trace")
    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.set_xlabel('Rank')
    ax0.set_ylabel('Frequency')
    popt, _ = curve_fit(zipf, rank, frequency, p0=[5.0], bounds=(1, np.inf))
    ax0.plot(rank, zipf(rank, *popt), label=f"Zipf[a={round(popt[0], 2)}]")
    ax0.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


# def plot_mae_dynamic_and_linked_cell(count_sketch: int, count_log: int):
#     packets_per_modify_size = COUNT_PACKETS // count_sketch
#     packets_per_log = COUNT_PACKETS // count_log
#
#     for i in range(count_sketch):
#         result_countmin = execute_command([
#             "--type", "countmin",
#             "--width", str(250 * (i + 1)),
#             "--depth", "4",
#             "--repeat", "log_average_relative_error", str(packets_per_log),
#         ])
#         x_countmin = np.array(result_countmin.index.to_numpy())
#         y_countmin = np.array(
#             result_countmin['log_average_relative_error'].to_numpy())
#         plt.figure(num="mae_dynamic_countmin")
#         plt.plot(x_countmin, y_countmin, label=f"countmin {i + 1}")
#         plt.figure(num="mae_dynamic_countmin_derivative")
#         plt.plot(x_countmin, np.gradient(
#             y_countmin, packets_per_log), label=f"countmin {i + 1}")
#
#     result_dynamic = execute_command([
#         "--type", "cellsketch",
#         "--width", "250",
#         "--depth", "4",
#         "--repeat", "expand", str(packets_per_modify_size),
#         "--repeat", "log_average_relative_error", str(packets_per_log)])
#
#     x_dynamic = np.array(result_dynamic.index.to_numpy())
#     y_dynamic = np.array(result_dynamic['log_average_relative_error'].to_numpy())
#     plt.figure(num="mae_dynamic_countmin")
#     plt.plot(x_dynamic, y_dynamic, label="dynamic")
#     plt.figure(num="mae_dynamic_countmin_derivative")
#     plt.plot(x_dynamic, np.gradient(
#         y_dynamic, packets_per_log), label=f"dynamic")
#
#     for xc in range(packets_per_modify_size, packets_per_modify_size * count_sketch, packets_per_modify_size):
#         plt.figure(num="mae_dynamic_countmin")
#         plt.axvline(x=xc, color='r', linestyle='dashed')
#         plt.figure(num="mae_dynamic_countmin_derivative")
#         plt.axvline(x=xc, color='r', linestyle='dashed')
#
#     for xc in range(packets_per_modify_size * count_sketch, COUNT_PACKETS, packets_per_modify_size):
#         plt.figure(num="mae_dynamic_countmin")
#         plt.axvline(x=xc, color='g', linestyle='dashed')
#         plt.figure(num="mae_dynamic_countmin_derivative")
#         plt.axvline(x=xc, color='g', linestyle='dashed')
#
#     plt.figure(num="mae_dynamic_countmin")
#     plt.legend()
#     plt.grid()
#     plt.ylabel('MAE')
#     plt.xlabel('packets')
#     plt.savefig('mae_dynamic_countmin.png')
#
#     plt.figure(num="mae_dynamic_countmin_derivative")
#     plt.legend()
#     plt.grid()
#     plt.ylabel("MAE'")
#     plt.xlabel('packets')
#     plt.savefig('mae_dynamic_countmin_derivative.png')
#
#     plt.close(fig)

def plot_branching_factor(branching_factors: list, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = max(branching_factors) * sketch_width
    packets_per_expand = COUNT_PACKETS // counters_added_per_row
    result_countmin = execute_command([
        "--type", "countmin",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--once", "log_average_relative_error", "0",
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
        "--once", "log_memory_usage", "0",
        "--repeat", "log_memory_usage", str(packets_per_log),
        "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
    ])
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
            "--repeat", "expand", str(packets_per_expand), "1",
            "--repeat", "compress", str(branching_factor), str("1000000"),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
        ])
        y_mae = np.array(
            result_cellsketch['log_average_relative_error'].to_numpy())
        y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
        x = np.array(result_cellsketch.index.to_numpy())
        ax0.plot(x, y_mem, label=f'GS-B{branching_factor}', marker=markers[i])
        ax1.plot(x, y_mae, label=f'GS-B{branching_factor}', marker=markers[i])
    ax0.legend()
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_skew_branching_factor(Bs: list, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    B_max = max(Bs)
    counters_added_per_row = B_max * sketch_width
    global COUNT_PACKETS
    OLD_COUNT_PACKETS = COUNT_PACKETS
    COUNT_PACKETS = 1000000
    packets_per_expand = COUNT_PACKETS / counters_added_per_row
    fig, ((ax0), (ax1)) = plt.subplots(1, 2, figsize=(8, 4))
    markers = ["D", "s", "^", ">", "v", "<", "*"]
    for i, B in enumerate(Bs):
        step = 0.01
        max_a = 2.2
        y_are_cmp = []
        y_are_uncmp = []
        x = []

        for a_float in np.arange(1.01, max_a, step):
            a = np.round(a_float, 2)
            packets_path = f"{filepath_zipf}\\zipf-{a}.txt"
            packets_per_expand_compressed = packets_per_expand * (B - 1) / B
            command_for_uncompressed = [
                "--repeat", "expand", str(round(packets_per_expand)), "1",
            ]
            command_for_compress = [
                "--repeat", "expand", str(round(packets_per_expand_compressed)), "1",
                "--repeat", "compress", str(B), str("1000000"),
            ]
            command_gs = [
                "--type", "cellsketch",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(B),
                "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
            ]
            result_gs_uncmp = execute_command(command_gs + command_for_uncompressed, packets_path)
            result_gs_cmp = execute_command(command_gs + command_for_compress, packets_path)
            are_cmp = np.array(result_gs_cmp['log_average_relative_error'].to_numpy())[-1]
            are_uncmp = np.array(result_gs_uncmp['log_average_relative_error'].to_numpy())[-1]
            y_are_cmp.append(are_cmp)
            y_are_uncmp.append(are_uncmp)
            x.append(a)
        ax0.plot(x, y_are_cmp, label=f'GS-B{B}', marker=markers[i])
        ax1.plot(x, y_are_uncmp, label=f'GS-B{B}', marker=markers[i])
    ax0.set_title("compressed")
    ax1.set_title("uncompressed")
    for ax in [ax0, ax1]:
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax.set_yscale('log')
        ax.grid()
        ax.set_ylabel('ARE')
        ax.set_xlabel('Zipfian Parameter')
        ax.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)
    COUNT_PACKETS = OLD_COUNT_PACKETS


def plot_gs_cms_comparison(B: int, L: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = sketch_width * (B ** (L - 1) - 1) / (B - 1) - sketch_width
    packets_per_expand = COUNT_PACKETS / counters_added_per_row

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["D", "s", "^"]
    for l in range(L):
        cm_width = sketch_width * B ** l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1)])
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
        "--repeat", "expand", str(packets_per_expand), "1",
        #        "--repeat", "compress", str(packets_per_expand), str(1000000),
        "--once", "log_average_relative_error", "0",
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
        "--once", "log_memory_usage", "0",
        "--repeat", "log_memory_usage", str(packets_per_log),
        "--once", "log_memory_usage", str(COUNT_PACKETS - 1)])
    y_mae = np.array(
        result_cellsketch['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_cellsketch['memory_usage'].to_numpy())
    x = np.array(result_cellsketch.index.to_numpy())
    ax0.plot(x, y_mem, label=f'GS-B{B}', marker='o')
    ax1.plot(x, y_mae, label=f'GS-B{B}', marker='o')
    ax0.legend()
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_cms_derivative_comparison(B: int, L: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added_per_row = sketch_width * (B ** L - 1) / (B - 1) - sketch_width
    packets_per_expand = math.floor(COUNT_PACKETS / counters_added_per_row)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    markers = ["D", "s", "^", "o", "p"]
    for l in range(L):
        cm_width = sketch_width * B ** l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
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
        "--repeat", "expand", str(packets_per_expand), "1",
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
    ax0.plot(x, y_mem, label=f'GS-B{B}', marker="o")
    ax1.plot(x, y_mae, label=f'GS-B{B}', marker="o")
    ax2.plot(x, d_mae, label=f"GS-B{B}", marker="o")
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_cms_static_comparison(B: int, L: int, count_log: int, figure_name):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["s", "p", "d", "D", "v", "^", ">", "<"]
    for l in range(1, L):
        cm_width = sketch_width * B ** l
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
        ])
        y_mae = result_countmin['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_countmin['memory_usage'].dropna().to_numpy()
        x = result_countmin.index.to_numpy()
        ax0.plot(x, y_mem, label=f"CMS-{l}", marker=markers[l])
        ax1.plot(x, y_mae, label=f"CMS-{l}", marker=markers[l])

        counters_added_per_row = int(sketch_width * (B ** (l + 1) - 1) / (B - 1) - sketch_width)
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
        x = result_cellsketch.index.to_numpy()
        ax0.plot(x, y_mem, label=f'GS-B{B}-L{l}', marker=markers[L + l])
        ax1.plot(x, y_mae, label=f'GS-B{B}-L{l}', marker=markers[L + l])

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
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_dcms_comparison(B: int, K: int, N: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["v", "^", ">", "<", "D", "o", '*']
    strategies = [
        lambda j: K * j,
        lambda j: K ** j,
    ]
    for i, strategy in enumerate(strategies):
        command_dcms = [
            "--type", "dynamic",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
        ]
        command_gs = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
        ]
        j = 1
        total = 0
        command_gs_big_alloc = command_gs.copy()
        while True:
            v = strategy(j)
            thresh = N * v
            total += thresh
            appended_sketch_width = sketch_width * v
            appended_counters_count = appended_sketch_width * B / (B - 1)
            if total >= COUNT_PACKETS:
                break
            compress_command = ["--once", "compress", str(total), "1000000000"]
            command_gs_big_alloc += ["--once", "expand", str(total), str(appended_sketch_width)]
            command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
            command_gs += ["--once", "expand", str(total), str(appended_counters_count)] + compress_command
            j += 1

        todo = [(0, "DCMS", command_dcms), (1, "GS", command_gs)]
        if i == 1:
            todo.append((2, "GS*", command_gs_big_alloc))
        for args in todo:
            s, type, command = args
            result = execute_command(command)
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            x = result.index.to_numpy()
            ax0.plot(x, y_mem, label=f"{type}-B{i}", marker=markers[i + s * 2])
            ax1.plot(x, y_mae, label=f"{type}-B{i}", marker=markers[i + s * 2])

    ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax0.grid()
    ax0.set_ylabel('Memory Usage (Bytes)')
    ax0.set_xlabel('Packets Count')
    ax1.grid()
    ax1.set_ylabel('ARE')
    ax1.set_xlabel('Packets Count')
    ax0.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_dcms_granular_comparison(B: int, K: int, N: int, granularity: int, count_log: int):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["s", "p", "d", "D", "v", "^", ">", "<"]
    strategies = [
        lambda j: K * j,
        lambda j: K ** j,
    ]
    for i, strategy in enumerate(strategies):
        command_dcms = [
            "--type", "dynamic",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
        ]
        command_gs = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1)
        ]
        j = 1
        total = 0
        while True:
            v = strategy(j)
            prev_total = total
            total += N * v
            appended_sketch_width = sketch_width * v
            total_appended_counters_count = appended_sketch_width * B / (B - 1)
            if total >= COUNT_PACKETS:
                break
            command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
            for g in range(1, granularity):
                index = prev_total + g / granularity * (total - prev_total)
                appended_counters_count = total_appended_counters_count / granularity
                compress_command = ["--once", "compress", str(index), "1000000000"]
                command_gs += ["--once", "expand", str(index), str(appended_counters_count)] + compress_command
            j += 1

        for args in [(0, "DCMS", command_dcms), (1, "GS", command_gs)]:
            s, type, command = args
            result = execute_command(command)
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            d_mae = np.gradient(y_mae, packets_per_log)
            x = result.index.to_numpy()
            ax0.plot(x, y_mem, label=f"{type}-{i}", marker=markers[i + s * 2])
            ax1.plot(x, y_mae, label=f"{type}-{i}", marker=markers[i + s * 2])

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
    plt.close(fig)


def plot_gs_undo_expand(B: int, L: int, max_cycles: int, granularity: int,
                        count_log_memory: int, count_log_are: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log_are = COUNT_PACKETS // count_log_are
    packets_per_log_memory = COUNT_PACKETS // count_log_memory

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["o", "s", "^", "v", "D", "p", "^", ">", "<"]
    for cycle_counts in range(1, max_cycles + 1):
        counters_added_per_row = int(sketch_width * (B ** L - 1) / (B - 1) - sketch_width)
        command = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--repeat", "log_average_relative_error", str(packets_per_log_are),
            "--once", "log_memory_usage", "0",
            "--once", "log_average_relative_error", "0",
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--repeat", "log_memory_usage", str(packets_per_log_memory)
        ]

        for cycle in range(cycle_counts):
            counters_chunk = round(counters_added_per_row / granularity)
            packet_chunk = round((COUNT_PACKETS - 2) / cycle_counts)
            cycle_begin = int(packet_chunk * cycle)
            cycle_end = int(packet_chunk * (cycle + 1))
            cycle_mid = int((cycle_begin + cycle_end) / 2)
            commands = [
                ["--once", "expand", str(cycle_mid - int(packet_chunk / 2 * i / granularity)), str(counters_chunk)] for
                i in range(granularity)]
            commands += [
                ["--once", "shrink", str(cycle_end - int(packet_chunk / 2 * i / granularity)), str(counters_chunk)] for
                i in range(granularity)]
            command += sum(commands, [])

        result_cellsketch = execute_command(command)
        result_are = result_cellsketch['log_average_relative_error'].dropna()
        result_mem = result_cellsketch['memory_usage'].dropna()
        y_are = result_are.to_numpy()
        x_are = result_are.index.values
        y_mem = result_mem.to_numpy()
        x_mem = result_mem.index.values
        ax0.plot(x_mem, y_mem, label=f'GS-B{B}-C{cycle_counts}', marker=markers[cycle_counts - 1])
        ax1.plot(x_are, y_are, label=f'GS-B{B}-C{cycle_counts}', marker=markers[cycle_counts - 1])

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
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_dynamic_undo_comparison(B_count: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    for B_index in range(B_count):
        B = 2 ** (B_index + 1)
        B_max = 2 ** B_count
        L = math.floor(log10(B_max * (B - 1) + 1) / log10(B)) + 1
        packets_per_modify = math.floor(COUNT_PACKETS / L / 2)
        expand_sizes = [sketch_width * B ** l for l in range(1, L)]
        command_lines_expand = sum(
            [["--once", "expand", str((i + 1) * packets_per_modify), str(e)] for i, e in enumerate(expand_sizes)], [])
        command_lines_shrink = sum(
            [["--once", "shrink", str((i + 1) * packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in
             enumerate(reversed(expand_sizes))], [])
        command_lines_compress = sum(
            [["--once", "compress", str((i + 1) * packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in
             enumerate(reversed(expand_sizes))], [])
        command_countmin = [
                               "--type", "dynamic",
                               "--width", str(sketch_width),
                               "--depth", str(sketch_depth),
                               "--repeat", "log_average_relative_error", str(packets_per_log),
                               "--once", "log_memory_usage", "0",
                               "--once", "log_average_relative_error", "0",
                               "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
                               "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
                               "--repeat", "log_memory_usage", str(packets_per_log)
                           ] + command_lines_expand
        command_geometric = [
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
                            ] + command_lines_expand

        names_type = ["DCMS", "GS"]
        markers = ["v", ">", "o", "^", "D", "<", "P"]
        for j, command_lines0 in enumerate([command_countmin, command_geometric]):
            result = execute_command(
                command_lines0 + command_lines_shrink
            )
            y_mae = np.array(
                result['log_average_relative_error'].to_numpy())
            y_mem = np.array(result['memory_usage'].to_numpy())
            x = np.array(result.index.to_numpy())
            ax0.plot(x, y_mem, label=f'{names_type[j]}-B{B}', marker=markers[j + B_index * 2])
            ax1.plot(x, y_mae, label=f'{names_type[j]}-B{B}', marker=markers[j + B_index * 2])

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
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


# replace with undo expand
def plot_gs_dynamic_expand_comparison(B: int, L: int, count_log: int):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    packets_per_modify = math.floor(COUNT_PACKETS / L)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    expand_sizes = [sketch_width * B ** l for l in range(1, L)]
    command_lines_expand = sum(
        [["--once", "expand", str((i + 1) * packets_per_modify), str(e)] for i, e in enumerate(expand_sizes)], [])
    command_lines_compress = sum([["--once", "compress", str((i + 1) * packets_per_modify), str(e)] for i, e in
                                  enumerate(reversed(expand_sizes))], [])
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
    plt.close(fig)


def plot_gs_derivative(B: int, L: int, count_expand: int, count_log: int, figure_name: str):
    sketch_width = 272
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

    N = sketch_width * (B ** L - 1) / (B - 1)
    expand_functions = [
        lambda x: N * math.sqrt(x),
        lambda x: N * math.log2(x + 1),
        lambda x: N * (2 ** x - 1),
        lambda x: N * math.sin(math.sqrt(x) * math.pi / 2),
        lambda x: N * math.asin(x * x) * 2 / math.pi
    ]

    for i, expand_function in enumerate(expand_functions):
        command = [
            "--type", "cellsketch",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1)]
        expands = []
        for expand_index in range(1, count_expand):
            packet_index = expand_index * packets_per_expand
            expand_size = expand_function(expand_index / (count_expand - 1)) - sum(expands)
            expands.append(math.floor(expand_size))
            command += ["--once", "expand", str(packet_index), str(expand_size)]
            command += ["--once", "compress", str(packet_index), str(1000000)]
        print(expands)
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
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


if __name__ == "__main__":
    # plot_gs_dynamic_undo_comparison(3, 24, "plot_gs_dynamic_undo_comparison")
    # plot_mae_dynamic_and_linked_cell(4, 128)
    # plot_mae_cellsketch_expand(2, 128)
    # plot_update_query_throughput_gs
    # plot_gs_expand_undo_compress_throughput(8, 6, "fig_gs_expand_undo_compress_throughput")
    plot_gs_update_query_throughput(9, 7, "fig_gs_update_query_throughput")
    # plot_cms_update_query_throughput(5, 8, 8, "fig_cms_throughput")
    # plot_gs_dcms_comparison(2, 2, 640, 32, "fig_gs_dcms_comparison_big_allocate")
    # plot_gs_cms_derivative_comparison(2, 4, 16, "fig_gs_cms_derivative")
    # plot_gs_cms_comparison(2, 3, 16, "fig_gs_cms_compare")
    # plot_ip_distribution_zipf("fig_ip_distribution_zipf")
    # plot_ip_distribution("fig_ip_distribution")
    # plot_ip_distribution("fig_ip_distribution")
    # plot_branching_factor([2, 4, 8], 16, "fig_branching_factor")
    # plot_gs_derivative(2, 3, 256, 16, "fig_gs_derivative")
    # plot_gs_undo_expand(B=2, L=3, max_cycles=4, granularity=128, count_log_memory=24, count_log_are=24, figure_name="fig_gs_undo_expand")
    # plot_gs_cms_static_comparison(2, 4, 16, "fig_gs_cms_static_comparison")
    # plot_gs_dcms_granular_comparison(2, 2, 640, 4, 32)
    # plot_gs_skew_branching_factor([2, 4, 8, 12, 16], 16, "fig_gs_skew_branching_factor")
