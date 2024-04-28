import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import subprocess as sp
import multiprocessing as mp
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
    filepath_executable = "../cpp/DynamicSketch"

COUNT_PACKETS_MAX = 37700000
COUNT_PACKETS = min(37700000, COUNT_PACKETS_MAX)


def generate_dcms_expands_command(N: int, K: int, sketch_width: int, sketch_depth: int,
                                  strategy_name: str, is_geometric: bool, is_compressed: bool, B=None):
    strategy_funcs = {
        "linear": lambda x: K * x,
        "exponential": lambda x: K ** x
    }
    strategy = strategy_funcs[strategy_name]
    j = 1
    total = 0
    command = []
    while True:
        v = strategy(j)
        j += 1
        total += N * v
        appended_size = sketch_width * v
        if is_geometric:
            appended_size *= sketch_depth
        if total >= COUNT_PACKETS:
            break
        if not is_compressed:
            command_expand = ["--once", "expand", str(total), str(appended_size)]
            command.extend(command_expand)
        else:
            compress_command = ["--once", "compress", str(total), str(appended_size * B / (B - 1))]
            command_expand_compressed = ["--once", "expand", str(total), str(appended_size * B / (B - 1))]
            command.extend(command_expand_compressed + compress_command)
    return command


def calculate_layer_width(w, b, layer_index):
    return w * b ** layer_index


def width_for_expand(w, b, to_layer_index):
    return w * ((b ** (to_layer_index + 1) - 1) / (b - 1)) - w


def execute_command(arguments: list, packets_path=filepath_packets, count_packets=COUNT_PACKETS):
    command = [filepath_executable, "--limit_file",
               packets_path, str(count_packets)] + arguments
    print("executing:", ' '.join(command))
    result = sp.run(command, stdout=sp.PIPE,
                    stderr=sp.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise ValueError(
            f"command: {' '.join(command)} caused error: {result.stderr}")
    raw_output = result.stdout.replace("\n", "").replace("\t", "")[:-2] + ']'
    output_dict = json.loads(raw_output)
    output_df = pd.DataFrame(output_dict).groupby('index').filter(lambda x: True).set_index("index")
    return output_df


def get_packets(filepath_packets: str):
    file_packets = open(filepath_packets, "r")
    return np.array([int(ip) for ip in file_packets.readlines()])


def plot_cms_update_query_throughput(count_width: int, count_depth: int, depth: int, branching_factor: int,
                                     figure_name: str):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    packets_per_log_time = COUNT_PACKETS - 1
    throughputs_update = np.zeros((count_depth, count_width))
    throughputs_query = np.zeros((count_depth, count_width))
    for D in range(1, count_depth + 1):
        for sketch_width_pow in range(0, count_width):
            W = 272 * (2 ** sketch_width_pow)
            result = execute_command([
                "--type", "countmin",
                "--width", str(W),
                "--depth", str(D * depth),
                "--repeat", "log_update_time", str(packets_per_log_time),
                "--repeat", "log_query_time", str(packets_per_log_time)])
            ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            throughput_update = 1 / (ms_per_update * 1e3)
            ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughput_query = 1 / (ms_per_query * 1e3)
            throughputs_update[D - 1, sketch_width_pow] = throughput_update
            throughputs_query[D - 1, sketch_width_pow] = throughput_query

    im0 = ax0.imshow(throughputs_update, origin='lower',
                     extent=[1, count_depth + 1, 0, count_width])
    im1 = ax1.imshow(throughputs_query, origin='lower',
                     extent=[1, count_depth + 1, 0, count_width])
    ax0.set_yticklabels([i for i in range(depth, depth * (count_depth + 2), depth)])
    ax1.set_yticklabels([i for i in range(depth, depth * (count_depth + 2), depth)])

    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 0.5 + 1, j + 0.5, int(label), ha='center', va='center')

    ax0.set_title('CMS Update MOPS')
    ax0.set_ylabel('Depth')
    ax0.set_xlabel('log2(Width/W)')
    ax1.set_title('CMS Query MOPS')
    ax1.set_ylabel('Depth')
    ax1.set_xlabel('log2(Width/W)')
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
                "--type", "geometric",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--once", "log_update_time", str(COUNT_PACKETS - 1),
                "--once", "log_query_time", str(COUNT_PACKETS - 1)])

            ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            throughputs_update[l, b - 2] = 1 / (ms_per_update * 1e3)
            ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughputs_query[l, b - 2] = 1 / (ms_per_query * 1e3)

    im0 = ax0.imshow(throughputs_update, origin='lower', extent=[2, B, 0, L])
    im1 = ax1.imshow(throughputs_query, origin='lower', extent=[2, B, 0, L])
    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 0.5, int(label), ha='center', va='center')

    ax0.set_title('GS Update MOPS')
    ax0.set_xlabel('Branching Factor')
    ax0.set_ylabel('Layers')
    ax1.set_title('Uncompressed GS Query MOPS')
    ax1.set_xlabel('Branching Factor')
    ax1.set_ylabel('Layers')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)

def plot_gs_mh_update_query_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))
    throughputs_update = np.zeros((L, B - 2))
    throughputs_query = np.zeros((L, B - 2))
    for sketch_type, [_ax0, _ax1] in zip(["geometric", "multihash"], [[ax0, ax1], [ax2, ax3]]):
        for l in range(L):
            for b in range(2, B):
                expand_size = sketch_width * ((b ** (l + 1) - 1) / (b - 1)) - sketch_width
                result = execute_command([
                    "--type", sketch_type,
                    "--width", str(sketch_width),
                    "--depth", str(sketch_depth),
                    "--branching_factor", str(b),
                    "--once", "expand", "0", str(expand_size),
                    "--once", "log_update_time", str(COUNT_PACKETS - 1),
                    "--once", "log_query_time", str(COUNT_PACKETS - 1)])

                ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
                throughputs_update[l, b - 2] = 1 / (ms_per_update * 1e3)
                ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
                throughputs_query[l, b - 2] = 1 / (ms_per_query * 1e3)

        im0 = _ax0.imshow(throughputs_update, origin='lower', extent=[2, B, 0, L])
        im1 = _ax1.imshow(throughputs_query, origin='lower', extent=[2, B, 0, L])
        for (throughputs, ax) in [(throughputs_update, _ax0), (throughputs_query, _ax1)]:
            for (j, i), label in np.ndenumerate(throughputs):
                ax.text(i + 2.5, j + 0.5, int(label), ha='center', va='center')

    ax0.set_title('GS U MOPS')
    ax0.set_xlabel('B')
    ax0.set_ylabel('L')
    ax1.set_title('GS Q MOPS')
    ax1.set_xlabel('B')
    ax1.set_ylabel('L')
   
    ax2.set_title('MH GS U MOPS')
    ax2.set_xlabel('B')
    ax2.set_ylabel('L')
    ax3.set_title('MH GS Q MOPS')
    ax3.set_xlabel('B')
    ax3.set_ylabel('L')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_dcms_update_query_throughput(B: int, S: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    throughputs_update = np.zeros((S, B - 2))
    throughputs_query = np.zeros((S, B - 2))
    for s in range(1, S + 1):
        for b in range(2, B):
            command = [
                "--type", "dynamic",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--once", "log_update_time", str(COUNT_PACKETS - 1),
                "--once", "log_query_time", str(COUNT_PACKETS - 1)]
            for i in range(1, s):
                sketch_size = b ** i * sketch_width
                command += ["--once", "expand", "0", str(sketch_size)]
            result = execute_command(command)
            ms_per_update = np.array(result['log_update_time'].dropna().to_numpy()).mean()
            throughputs_update[s - 1, b - 2] = 1 / (ms_per_update * 1e3)
            ms_per_query = np.array(result['log_query_time'].dropna().to_numpy()).mean()
            throughputs_query[s - 1, b - 2] = 1 / (ms_per_query * 1e3)
    im0 = ax0.imshow(throughputs_update, origin='lower', extent=[2, B, 1, S + 1])
    im1 = ax1.imshow(throughputs_query, origin='lower', extent=[2, B, 1, S + 1])

    for (throughputs, ax) in [(throughputs_update, ax0), (throughputs_query, ax1)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 1.5, int(label), ha='center', va='center')

    ax0.set_title('DCMS Update MOPS')
    ax0.set_xlabel('Branching Factor')
    ax0.set_ylabel('Sketches')
    ax1.set_title('DCMS Query MOPS')
    ax1.set_xlabel('Branching Factor')
    ax1.set_ylabel('Sketches')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


# need to actually compress
def plot_gs_compress_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax_compress, ax_query) = plt.subplots(1, 2, figsize=(8, 4))
    count_log_time = 16
    count_query = 64
    throughputs_compress = np.zeros((L, B - 2))
    throughputs_query = np.zeros((L, B - 2))
    count_repeat = 10
    for l in range(1, L + 1):
        for b in range(2, B):
            expand_size = sketch_depth * width_for_expand(sketch_width, b, l)
            last_layer_size = sketch_depth * calculate_layer_width(sketch_width, b, l)
            one_before_last_layer_size = sketch_depth * calculate_layer_width(sketch_width, b, l - 1)
            pre_compress_size = max(0, expand_size - last_layer_size - one_before_last_layer_size)
            avg_ms_per_compress = 0
            for _ in range(count_repeat):
                result_compress = execute_command([
                    "--type", "geometric",
                    "--width", str(sketch_width),
                    "--depth", str(sketch_depth),
                    "--branching_factor", str(b),
                    "--once", "expand", "0", str(expand_size),
                    "--once", "compress", "1", str(pre_compress_size),
                    "--once", "compress", "2", str(expand_size),  # used expand size as "huge"
                ], count_packets=5)
                avg_ms_per_compress += np.array(result_compress['log_compress_time'].dropna().to_numpy())[-1]
            avg_ms_per_compress /= count_repeat

            result_query = execute_command([
                "--type", "geometric",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(b),
                "--once", "expand", "0", str(expand_size),
                "--once", "compress", "1", str(expand_size),
                "--once", "log_query_time", str(COUNT_PACKETS - 1)
            ])
            ms_per_query = np.array(result_query['log_query_time'].dropna().to_numpy())[-1]
            throughputs_compress[l - 1, b - 2] = log10(1 / (avg_ms_per_compress * 1e3))
            throughputs_query[l - 1, b - 2] = 1 / (ms_per_query * 1e3)
    im0 = ax_compress.imshow(throughputs_compress, origin='lower', extent=[2, B, 0, L])
    im1 = ax_query.imshow(throughputs_query, origin='lower', extent=(2, B, 1, L + 1))

    for (j, i), label in np.ndenumerate(throughputs_query):
        ax_query.text(i + 2.5, j + 1.5, int(label), ha='center', va='center')

    for (j, i), label in np.ndenumerate(throughputs_compress):
        ax_compress.text(i + 2.5, j + 0.5, int(label), ha='center', va='center')

    ax_compress.set_title('GS Compress Log10(MOPS)')
    ax_compress.set_xlabel('Branching Factor')
    ax_compress.set_ylabel('Layers')
    ax_query.set_title('GS Compressed Query MOPS')
    ax_query.set_xlabel('Branching Factor')
    ax_query.set_ylabel('Layers')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_expand_undo_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax_expand, ax_shrink) = plt.subplots(1, 2, figsize=(8, 4))
    count_log_time = 16
    count_query = 64
    packets_per_log_time = COUNT_PACKETS // count_log_time
    packets_per_query = COUNT_PACKETS // count_query
    throughputs_expand = np.zeros((L, B - 2))
    throughputs_undo = np.zeros((L, B - 2))
    count_repeat = 10
    for layer_index in range(L):
        for b in range(2, B):
            expand_size = sketch_depth * (sketch_width * ((b ** (layer_index + 1) - 1) / (b - 1)) - sketch_width)
            next_layer_width = sketch_depth * sketch_width * b ** (layer_index + 1)
            avg_ms_per_expand = 0
            avg_ms_per_shrink = 0
            for _ in range(count_repeat):
                result = execute_command([
                    "--type", "geometric",
                    "--width", str(sketch_width),
                    "--depth", str(sketch_depth),
                    "--branching_factor", str(b),
                    "--once", "expand", "0", str(expand_size),
                    "--once", "expand", "1", str(next_layer_width),
                    "--once", "shrink", "2", str(next_layer_width),
                ])
                avg_ms_per_expand += np.array(result['log_expand_time'].dropna().to_numpy())[-1]
                avg_ms_per_shrink += np.array(result['log_shrink_time'].dropna().to_numpy())[-1]
            avg_ms_per_expand /= count_repeat
            avg_ms_per_shrink /= count_repeat
            throughputs_expand[layer_index, b - 2] = log10(1 / (avg_ms_per_expand * 1e3))
            throughputs_undo[layer_index, b - 2] = log10(1 / (avg_ms_per_shrink * 1e3))
    im0 = ax_expand.imshow(throughputs_expand, origin='lower', extent=[2, B, 1, L + 1])
    im1 = ax_shrink.imshow(throughputs_undo, origin='lower', extent=[2, B, 1, L + 1])

    for (throughputs, ax) in [(throughputs_expand, ax_expand), (throughputs_undo, ax_shrink)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 1.5, int(label), ha='center', va='center')

    ax_expand.set_title('GS Expand Log10(MOPS)')
    ax_expand.set_xlabel('Branching Factor')
    ax_expand.set_ylabel('Layers')
    ax_shrink.set_title('GS Undo Log10(MOPS)')
    ax_shrink.set_xlabel('Branching Factor')
    ax_shrink.set_ylabel('Layers')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_dcms_undo_throughput(B: int, L: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5

    fig, (ax_gs, ax_dcms) = plt.subplots(1, 2, figsize=(8, 4))
    count_log_time = 16
    count_query = 64
    throughputs = {
        "geometric": np.zeros((L, B - 2)),
        "dynamic": np.zeros((L, B - 2))
    }
    count_repeat = 10
    for sketchtype in ["dynamic", "geometric"]:
        for layer_index in range(L):
            for b in range(2, B):
                arr_ms_per_expand = np.zeros(count_repeat)
                arr_ms_per_shrink = np.zeros(count_repeat)
                for c in range(count_repeat):
                    commmand = [
                        "--type", sketchtype,
                        "--width", str(sketch_width),
                        "--depth", str(sketch_depth),
                        "--branching_factor", str(b)
                    ]
                    # not a bug - DCMS gets expand size without depth (multiplies by depth) while GS doesnt
                    for i in range(1, layer_index + 1):
                        expand_size = sketch_width * (b ** i)
                        if sketchtype == "geometric":
                            expand_size *= sketch_depth
                        commmand.extend(["--once", "expand", str(i - 1), str(expand_size)])
                    commmand.extend(["--once", "expand", str(layer_index), str(sketch_width * b ** (layer_index + 1))])
                    commmand.extend(
                        ["--once", "shrink", str(layer_index + 1), str(sketch_width * b ** (layer_index + 1))])
                    result = execute_command(commmand, count_packets=100)
                    arr_ms_per_shrink[c] = np.array(result['log_shrink_time'].dropna().to_numpy())[-1]
                avg_ms_per_shrink = arr_ms_per_shrink.mean()
                throughputs[sketchtype][layer_index, b - 2] = log10(1 / (avg_ms_per_shrink * 1e3))
    im0 = ax_gs.imshow(throughputs["geometric"], origin='lower', extent=[2, B, 1, L + 1])
    im1 = ax_dcms.imshow(throughputs["dynamic"], origin='lower', extent=[2, B, 1, L + 1])

    for (throughputs, ax) in [(throughputs["geometric"], ax_gs), (throughputs["dynamic"], ax_dcms)]:
        for (j, i), label in np.ndenumerate(throughputs):
            ax.text(i + 2.5, j + 1.5, int(label), ha='center', va='center')

    ax_gs.set_title('GS Undo Log10(MOPS)')
    ax_gs.set_xlabel('Branching Factor')
    ax_gs.set_ylabel('Layers')
    ax_dcms.set_title('DCMS Undo Log10(MOPS)')
    ax_dcms.set_xlabel('Branching Factor')
    ax_dcms.set_ylabel('Layers')
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


def plot_branching_factor(branching_factors: list, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log
    counters_added = max(branching_factors) * sketch_width * sketch_depth
    packets_per_expand = COUNT_PACKETS // counters_added
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
    y_mae = np.array(result_countmin['log_average_relative_error'].dropna().to_numpy())
    y_mem = np.array(result_countmin['memory_usage'].dropna().to_numpy())
    x = np.array(result_countmin.index.to_numpy())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax0.plot(x, y_mem, label="CMS-L0", marker='o')
    ax1.plot(x, y_mae, label="CMS-L0", marker='o')
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
        result_geometric = execute_command([
            "--type", "geometric",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(branching_factor),
            "--repeat", "expand", str(packets_per_expand), "1",
            "--repeat", "compress", str(packets_per_expand), "10000",
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
        ])
        y_mae = result_geometric['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_geometric['memory_usage'].dropna().to_numpy()
        x_mae = result_geometric['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result_geometric['memory_usage'].dropna().index.to_numpy()
        ax0.plot(x_mem, y_mem, label=f'GS-B{branching_factor}', marker=markers[i])
        ax1.plot(x_mae, y_mae, label=f'GS-B{branching_factor}', marker=markers[i])
    ax0.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_skew_branching_factor(Bs: list, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    B_max = max(Bs)
    counters_added = sketch_depth * width_for_expand(sketch_width, B_max, 1)
    COUNT_PACKETS_SYNTH = 1000000
    packets_per_expand = COUNT_PACKETS_SYNTH / counters_added
    fig, ((ax0), (ax1)) = plt.subplots(1, 2, figsize=(8, 4))
    markers = ["D", "s", "^", ">", "v", "<", "*"]
    min_step = 0.01
    max_a = 2.29

    for i, B in enumerate(Bs):
        step = min_step
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
                "--repeat", "compress", str(round(packets_per_expand_compressed)), str(COUNT_PACKETS_SYNTH),
            ]
            command_gs = [
                "--type", "geometric",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(B),
                "--once", "log_average_relative_error", str(COUNT_PACKETS_SYNTH - 1)
            ]
            result_gs_uncmp = execute_command(command_gs + command_for_uncompressed, packets_path,
                                              count_packets=COUNT_PACKETS_SYNTH)
            result_gs_cmp = execute_command(command_gs + command_for_compress, packets_path,
                                            count_packets=COUNT_PACKETS_SYNTH)
            are_cmp = result_gs_cmp['log_average_relative_error'].dropna().to_numpy()[-1]
            are_uncmp = result_gs_uncmp['log_average_relative_error'].dropna().to_numpy()[-1]
            print(a, are_cmp, are_uncmp)
            y_are_cmp.append(are_cmp)
            y_are_uncmp.append(are_uncmp)
            x.append(a)
        count_a = len(np.arange(1.01, max_a, step))
        plot_every = max(1, count_a // count_log)
        ax0.plot(x, y_are_cmp, label=f'GS-B{B}', marker=markers[i], markevery=plot_every)
        ax1.plot(x, y_are_uncmp, label=f'GS-B{B}', marker=markers[i], markevery=plot_every)
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


def plot_gs_cms_derivative_comparison(B: int, L: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    counters_added = (sketch_depth * sketch_width * (B ** L - 1) / (B - 1) - sketch_width * sketch_depth)
    packets_per_expand = math.floor(COUNT_PACKETS / counters_added)
    assert packets_per_expand > 0
    APPROX_COUNT_PACKETS = counters_added * packets_per_expand
    packets_per_log = APPROX_COUNT_PACKETS // count_log

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 4))

    markers = ["D", "s", "^", "o", "p"]
    for l in range(L):
        cm_width = sketch_width * (B ** l)
        result_countmin = execute_command([
            "--type", "countmin",
            "--width", str(cm_width),
            "--depth", str(sketch_depth),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(APPROX_COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(APPROX_COUNT_PACKETS - 1)
        ], count_packets=APPROX_COUNT_PACKETS)
        y_mae = result_countmin['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_countmin['memory_usage'].dropna().to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        x_mae = result_countmin['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result_countmin['memory_usage'].dropna().index.to_numpy()
        ax0.plot(x_mem, y_mem, label=f"CMS-L{l}", marker=markers[l])
        ax1.plot(x_mae, y_mae, label=f"CMS-L{l}", marker=markers[l])
        ax2.plot(x_mae, d_mae, label=f"CMS-L{l}", marker=markers[l])

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
        "--type", "geometric",
        "--width", str(sketch_width),
        "--depth", str(sketch_depth),
        "--branching_factor", str(B),
        "--repeat", "expand", str(packets_per_expand), "1",
        "--repeat", "compress", str(packets_per_expand), "1000000",
        "--repeat", "log_average_relative_error", str(packets_per_log),
        "--once", "log_memory_usage", "0",
        "--once", "log_average_relative_error", "0",
        "--once", "log_memory_usage", str(APPROX_COUNT_PACKETS - 1),
        "--once", "log_average_relative_error", str(APPROX_COUNT_PACKETS - 1),
        "--repeat", "log_memory_usage", str(packets_per_log),
    ]
    result_geometric = execute_command(command, count_packets=APPROX_COUNT_PACKETS)
    y_mae = result_geometric['log_average_relative_error'].dropna().to_numpy()
    y_mem = result_geometric['memory_usage'].dropna().to_numpy()
    d_mae = np.gradient(y_mae, packets_per_log)
    x_mae = result_geometric['log_average_relative_error'].dropna().index.to_numpy()
    x_mem = result_geometric['memory_usage'].dropna().index.to_numpy()
    ax0.plot(x_mem, y_mem, label=f'GS-B{B}', marker="o")
    ax1.plot(x_mae, y_mae, label=f'GS-B{B}', marker="o")
    ax2.plot(x_mae, d_mae, label=f"GS-B{B}", marker="o")
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
        cm_width = sketch_width * (B ** l)
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
        x_mae = result_countmin['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result_countmin['memory_usage'].dropna().index.to_numpy()
        ax0.plot(x_mem, y_mem, label=f"CMS-L{l}", marker=markers[l])
        ax1.plot(x_mae, y_mae, label=f"CMS-L{l}", marker=markers[l])

        counters_added = int(sketch_width * (B ** (l + 1) - 1) / (B - 1) - sketch_width) * sketch_depth
        command = [
            "--type", "geometric",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "expand", "0", str(counters_added),
            "--once", "compress", "0", str(counters_added),
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_memory_usage", "0",
            "--once", "log_average_relative_error", "0",
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--repeat", "log_memory_usage", str(packets_per_log),
        ]
        result_geometric = execute_command(command)
        y_mae = result_geometric['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_geometric['memory_usage'].dropna().to_numpy()
        x_mae = result_geometric['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result_geometric['memory_usage'].dropna().index.to_numpy()
        ax0.plot(x_mem, y_mem, label=f'GS-B{B}-L{l}', marker=markers[L + l])
        ax1.plot(x_mae, y_mae, label=f'GS-B{B}-L{l}', marker=markers[L + l])

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


def plot_gs_dcms_comparison(B: int, K: int, N: int, count_log: int, figure_name: str):
    sketch_width = 50
    sketch_depth = 2
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["v", "^", ">", "<", "D", "o", '*']
    strategies = [
        "linear",
        "exponential"
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
            "--type", "geometric",
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

        command_dcms += generate_dcms_expands_command(N, K, sketch_width, sketch_depth, strategy, False, False, B)
        command_gs_compressed = command_gs + generate_dcms_expands_command(N, K, sketch_width, sketch_depth, strategy,
                                                                           True, True, B)
        command_gs_uncompressed = command_gs + generate_dcms_expands_command(N, K, sketch_width, sketch_depth, strategy,
                                                                             True, False, B)
        todo = [(0, "DCMS", command_dcms), (1, "GS-C", command_gs_compressed), (2, "GS-U", command_gs_uncompressed)]
        for args in todo:
            s, type, command = args
            result = execute_command(command)
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            x_mae = result['log_average_relative_error'].dropna().index.to_numpy()
            x_mem = result['memory_usage'].dropna().index.to_numpy()
            strategy_char = "LIN" if i == 0 else "EXP"
            ax0.plot(x_mem, y_mem, label=f"{type}-{strategy_char}", marker=markers[s + i * 3])
            ax1.plot(x_mae, y_mae, label=f"{type}-{strategy_char}", marker=markers[s + i * 3])

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


def plot_dcms_memory_usage(KS: list, NS: list, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax) = plt.subplots(1, 1, figsize=(8, 4))

    linestyles = ["dashed", "dashdot", "dotted"]
    colors = ["olive", "purple"]
    strategy_names = ["LIN", "EXP"]
    markers = ["v", "^", ">", "<", "D", "o", '*']
    for kn_index, (K, N) in enumerate(zip(KS, NS)):
        strategies = [
            lambda v: K * v,
            lambda v: K ** v,
        ]

        bytes_per_int = 4
        slope = bytes_per_int * sketch_width * sketch_depth / N
        ax.plot([0, COUNT_PACKETS], [0, COUNT_PACKETS * slope],
                color=colors[kn_index], linestyle=linestyles[kn_index],
                label=f"SLP-N{N}-K{K}", linewidth=3)

        for i, strategy in enumerate(strategies):
            command_dcms = [
                "--type", "dynamic",
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--once", "log_memory_usage", "0",
                "--repeat", "log_memory_usage", str(packets_per_log),
                "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
                "--once", "log_average_relative_error", "0",
            ]
            j = 1
            total = 0
            while True:
                v = strategy(j)
                thresh = N * v
                total += thresh
                appended_sketch_width = sketch_width * v
                if total >= COUNT_PACKETS:
                    break
                command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
                j += 1

            result = execute_command(command_dcms)
            y_mem = result['memory_usage'].dropna().to_numpy()
            x = result['memory_usage'].dropna().index.to_numpy()
            ax.plot(x, y_mem, label=f"{strategy_names[i]}-N{N}-K{K}", marker=markers[kn_index * 2 + i])

    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.grid()
    ax.set_ylabel('Memory Usage (Bytes)')
    ax.set_xlabel('Packets Count')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


def plot_gs_dcms_granular_comparison(B: int, K: int, N: int, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))

    markers = ["v", "^", ">", "<", "D", "o", '*']
    strategies = [
        lambda v: K * v,
        lambda v: K ** v,
    ]

    for i, is_compressed in enumerate([True, False]):
        packets_per_expand = N / (sketch_depth * sketch_width)
        if is_compressed:
            packets_per_expand *= ((B - 1) / B)

        command_gs = [
            "--type", "geometric",
            "--width", str(sketch_width),
            "--depth", str(sketch_depth),
            "--branching_factor", str(B),
            "--once", "log_memory_usage", "0",
            "--repeat", "log_memory_usage", str(packets_per_log),
            "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
            "--once", "log_average_relative_error", "0",
            "--repeat", "log_average_relative_error", str(packets_per_log),
            "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
            "--repeat", "expand", str(packets_per_expand), "1"
        ]
        if is_compressed:
            command_gs += ["--repeat", "compress", str(packets_per_expand), str(COUNT_PACKETS)]

        result = execute_command(command_gs)
        y_mae = result['log_average_relative_error'].dropna().to_numpy()
        y_mem = result['memory_usage'].dropna().to_numpy()
        x_mae = result['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result['memory_usage'].dropna().index.to_numpy()
        compress_char = "C" if is_compressed else "U"
        ax0.plot(x_mem, y_mem, label=f"GS-B{B}-{compress_char}", marker=markers[-i])
        ax1.plot(x_mae, y_mae, label=f"GS-B{B}-{compress_char}", marker=markers[-i])

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

        j = 1
        total = 0
        while True:
            v = strategy(j)
            thresh = N * v
            total += thresh
            appended_sketch_width = sketch_width * v
            if total >= COUNT_PACKETS:
                break
            command_dcms += ["--once", "expand", str(total), str(appended_sketch_width)]
            j += 1

        result = execute_command(command_dcms)
        y_mae = result['log_average_relative_error'].dropna().to_numpy()
        y_mem = result['memory_usage'].dropna().to_numpy()
        x_mae = result['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result['memory_usage'].dropna().index.to_numpy()
        str_char = "LIN" if i == 0 else "EXP"
        ax0.plot(x_mem, y_mem, label=f"DCMS-{str_char}", marker=markers[i])
        ax1.plot(x_mae, y_mae, label=f"DCMS-{str_char}", marker=markers[i])

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

def get_unique_packets_count(count_include, filepath = filepath_packets):
    packets = get_packets(filepath)
    return len(Counter(packets[:count_include]).keys())

def plot_gs_error_heavy_hitters(B: int, K: int, N: int, hh_percent_min: float, hh_percent_max: float, count_log: int, figure_name: str):
    sketch_width = 272
    sketch_depth = 5
    packets_per_log = COUNT_PACKETS // count_log

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    j = 0
    unique_packets_count = get_unique_packets_count(COUNT_PACKETS)
    max_hh = math.ceil(unique_packets_count * hh_percent_max)
    min_hh = math.floor(unique_packets_count * hh_percent_min)
    markers = ["v", "^", ">", "<", "D", "o", '*']
    for strategy_name in ["exponential", "linear"]:
        for sketchtype in ["geometric_uncompressed", "geometric_compressed", "dynamic"]:
            real_sketchtype = sketchtype.split('_')[0]
            
            command = [
                "--type", real_sketchtype,
                "--width", str(sketch_width),
                "--depth", str(sketch_depth),
                "--branching_factor", str(B),
                "--once", "log_memory_usage", "0",
                "--repeat", "log_memory_usage", str(packets_per_log),
                "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
                "--once", "log_heavy_hitters", str(COUNT_PACKETS - 1), str(max_hh)
            ]
            is_geometric = (real_sketchtype == "geometric")
            compress = sketchtype.startswith("geometric") and sketchtype.split('_')[1] == "compressed"
            command += generate_dcms_expands_command(N, K, sketch_width, sketch_depth, strategy_name, is_geometric,
                                                     compress, B)

            result = execute_command(command)
            heavy_hitters = result['log_heavy_hitters'].dropna().to_numpy()[0]
            absolute_errors = np.array([abs(e['count'] - e['query']) for e in heavy_hitters])
            y_aae = [0]
            x_aae = np.arange(len(absolute_errors))
            for i in range(len(absolute_errors)):
                y_aae.append(y_aae[-1] + absolute_errors[i])
            y_aae = y_aae[1:]
            for i in range(1, len(absolute_errors)):
                y_aae[i] = y_aae[i] / i
            y_are = []
            relative_errors = [abs(e['count'] - e['query']) / e['count'] for e in heavy_hitters]
            for i in range(1, len(relative_errors) + 1):
                r_es = relative_errors[:i]
                cur_are = np.array(r_es).mean()
                y_are.append(cur_are)
            # when the number of heavy hitters is small, it adds lots of noise to the begining of the graph - so we start from 100
            y_are = y_are[min_hh:]
            x_aae = x_aae[min_hh:]
            y_aae = y_aae[min_hh:]
            x_aae = x_aae / unique_packets_count
            y_mem = result['memory_usage'].dropna().to_numpy()
            x_mem = result['memory_usage'].dropna().index.to_numpy()
            label_sketchtype_dict = {
                "geometric_uncompressed" : "GS",
                "geometric_compressed" : "GS",
                "dynamic" : "DCMS"
            }
            label_compressed_dict = {
                "geometric_uncompressed" : "U-",
                "geometric_compressed" : "C-",
                "dynamic" : ""
            }
            label_strategy_dict = {
                "linear" : "LIN",
                "exponential" : "EXP"
            }
            label_name = f"{label_sketchtype_dict[sketchtype]}-{label_compressed_dict[sketchtype]}{label_strategy_dict[strategy_name]}"
            ax0.plot(x_aae, y_aae, 
                     label=label_name, 
                     marker=markers[j],
                     markevery=len(y_aae) // count_log
                     )
            ax1.plot(x_aae, y_are, 
                     label=label_name, 
                     marker=markers[j],
                    markevery=len(y_aae) // count_log
                    )
            j += 1
            # ax2.plot(x_mem, y_mem, label=f"{sketchtype}-B{B}-{strategy_name}", alpha=0.5, linewidth=5)
    ax0.set_yscale('log')
    #ax0.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax0.grid()
    ax0.set_xlabel('Percentage Heavy Hitters Included')
    ax0.set_ylabel('AAE')
    ax1.grid()
    ax1.set_xlabel('Percentage Heavy Hitters Included')
    ax1.set_ylabel('ARE')
    ax1.legend()
    # ax2.set_xlabel('Packets')
    # ax2.set_ylabel('Memory Usage')
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
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
        counters_added = int(sketch_width * (B ** L - 1) / (B - 1) - sketch_width) * sketch_depth
        command = [
            "--type", "geometric",
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
            counters_chunk = round(counters_added / granularity)
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

        result_geometric = execute_command(command)
        result_are = result_geometric['log_average_relative_error'].dropna()
        result_mem = result_geometric['memory_usage'].dropna()
        y_are = result_are.to_numpy()
        x_are = result_are.index.values
        y_mem = result_mem.to_numpy()
        x_mem = result_mem.index.values
        ax0.plot(x_mem, y_mem, label=f'GS-B{B}-F{cycle_counts}', marker=markers[cycle_counts - 1])
        ax1.plot(x_are, y_are, label=f'GS-B{B}-F{cycle_counts}', marker=markers[cycle_counts - 1])

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
        expand_sizes_dynamic = [sketch_width * B ** l for l in range(1, L)]
        expand_sizes_geometric = [sketch_depth * es for es in expand_sizes_dynamic]
        command_lines_expand_dynamic = sum(
            [["--once", "expand", str((i + 1) * packets_per_modify), str(e)] for i, e in
             enumerate(expand_sizes_dynamic)], [])
        command_lines_shrink_dynamic = sum(
            [["--once", "shrink", str((i + 1) * packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in
             enumerate(reversed(expand_sizes_dynamic))], [])
        command_lines_expand_geometric = sum(
            [["--once", "expand", str((i + 1) * packets_per_modify), str(e)] for i, e in
             enumerate(expand_sizes_geometric)], [])
        command_lines_shrink_geometric = sum(
            [["--once", "shrink", str((i + 1) * packets_per_modify + COUNT_PACKETS // 2), str(e)] for i, e in
             enumerate(reversed(expand_sizes_geometric))], [])
        command_countmin = [
                               "--type", "dynamic",
                               "--dcms_same_seed", "1",
                               "--width", str(sketch_width),
                               "--depth", str(sketch_depth),
                               "--repeat", "log_average_relative_error", str(packets_per_log),
                               "--once", "log_memory_usage", "0",
                               "--once", "log_average_relative_error", "0",
                               "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
                               "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
                               "--repeat", "log_memory_usage", str(packets_per_log)
                           ] + command_lines_expand_dynamic
        command_geometric = [
                                "--type", "geometric",
                                "--width", str(sketch_width),
                                "--depth", str(sketch_depth),
                                "--branching_factor", str(B),
                                "--repeat", "log_average_relative_error", str(packets_per_log),
                                "--once", "log_memory_usage", "0",
                                "--once", "log_average_relative_error", "0",
                                "--once", "log_memory_usage", str(COUNT_PACKETS - 1),
                                "--once", "log_average_relative_error", str(COUNT_PACKETS - 1),
                                "--repeat", "log_memory_usage", str(packets_per_log)
                            ] + command_lines_expand_geometric

        names_type = ["DCMS", "GS"]
        markers = ["v", ">", "o", "^", "D", "<", "P"]
        for j, command_lines0, command_lines_shrink in [(0, command_countmin, command_lines_shrink_dynamic),
                                                        (1, command_geometric, command_lines_shrink_geometric)]:
            result = execute_command(
                command_lines0 + command_lines_shrink
            )
            y_mae = result['log_average_relative_error'].dropna().to_numpy()
            y_mem = result['memory_usage'].dropna().to_numpy()
            x_mae = result['log_average_relative_error'].dropna().index.to_numpy()
            x_mem = result['memory_usage'].dropna().index.to_numpy()
            ax0.plot(x_mem, y_mem, label=f'{names_type[j]}-B{B}', marker=markers[j + B_index * 2])
            ax1.plot(x_mae, y_mae, label=f'{names_type[j]}-B{B}', marker=markers[j + B_index * 2])

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

    expand_sizes = [sketch_width * (B ** l) for l in range(1, L)]
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
                            "--type", "geometric",
                            "--width", str(sketch_width),
                            "--depth", str(sketch_depth),
                            "--branching_factor", str(B),
                            "--repeat", "log_average_relative_error", str(packets_per_log),
                            "--repeat", "log_memory_usage", str(packets_per_log)
                        ] + command_lines_expand + command_lines_compress
    result_geometric = execute_command(
        command_geometric
    )
    y_mae = np.array(
        result_geometric['log_average_relative_error'].to_numpy())
    y_mem = np.array(result_geometric['memory_usage'].to_numpy())
    x = np.array(result_geometric.index.to_numpy())
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

    N = sketch_depth * sketch_width * ((B ** L - 1) / (B - 1) - 1)
    expand_functions = [
        lambda x: N * math.sin(math.sqrt(x) * math.pi / 2),
        lambda x: N * math.sqrt(x),
        lambda x: N * math.log2(x + 1),
        lambda x: N * (2 ** x - 1),
        lambda x: N * math.asin(x * x) * 2 / math.pi
    ]

    for i, expand_function in enumerate(expand_functions):
        command = [
            "--type", "geometric",
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
        for expand_index in range(1, count_expand + 1):
            packet_index = expand_index * packets_per_expand
            expand_size = expand_function(expand_index / count_expand) - sum(expands)
            expands.append(math.floor(expand_size))
            command += ["--once", "expand", str(packet_index), str(expand_size)]
        result_geometric = execute_command(command)
        y_mae = result_geometric['log_average_relative_error'].dropna().to_numpy()
        y_mem = result_geometric['memory_usage'].dropna().to_numpy()
        x_mae = result_geometric['log_average_relative_error'].dropna().index.to_numpy()
        x_mem = result_geometric['memory_usage'].dropna().index.to_numpy()
        d_mae = np.gradient(y_mae, packets_per_log)
        ax0.plot(x_mem, y_mem, label=f"$f_{i}$", marker=markers[i])
        ax1.plot(x_mae, y_mae, label=f"$f_{i}$", marker=markers[i])
        ax2.plot(x_mae, d_mae, label=f"$f_{i}$", marker=markers[i])
    ax1.legend()
    fig.tight_layout()
    plt.savefig(f'figures/{figure_name}')
    plt.close(fig)


font = {'family': 'sans-serif',
        'size': 12}

plt.rc('font', **font)


def parallel():
    funcs = [
        plot_gs_dynamic_undo_comparison,
        plot_gs_dcms_comparison,
        plot_gs_cms_derivative_comparison,
        plot_gs_derivative,
        plot_gs_undo_expand,
        plot_gs_cms_static_comparison,
        plot_gs_dcms_granular_comparison,
        plot_gs_skew_branching_factor,
        plot_dcms_memory_usage,
        plot_branching_factor,
        plot_ip_distribution_zipf,
        plot_ip_distribution,
        plot_gs_error_heavy_hitters
    ]

    args = [
        (3, 24, "plot_gs_dynamic_undo_comparison",),
        (2, 2, 2 * 5 * 272 * 100, 32, "fig_gs_dcms_comparison",),
        (2, 4, 16, "fig_gs_cms_derivative",),
        (2, 3, 256, 16, "fig_gs_derivative",),
        (2, 3, 4, 128, 24, 24, "fig_gs_undo_expand",),
        (2, 4, 16, "fig_gs_cms_static_comparison",),
        (2, 4, 2 * 5 * 272 * 100, 32, "fig_gs_dcms_granular_comparison",),
        ([2, 4, 8, 12, 16], 16, "fig_gs_skew_branching_factor",),
        ([2, 5], [1000, 500], 32, "fig_dcms_memory_usage",),
        ([2, 4, 8], 16, "fig_branching_factor",),
        ("fig_ip_distribution_zipf",),
        ("fig_ip_distribution",),
        (2, 2, 2 * 5 * 272 * 100, 0.001, 0.1, 16, "fig_gs_dcms_heavyhitters_error",),
    ]

    processes = [mp.Process(target=func, args=args) for (func, args) in zip(funcs, args)]
    [p.start() for p in processes]
    [p.join() for p in processes]


def serial():
    plot_gs_error_heavy_hitters(2, 2, 2 * 5 * 272 * 100, 0.001, 0.1, 16, "fig_gs_dcms_heavyhitters_error")
    plot_gs_dynamic_undo_comparison(3, 24, "plot_gs_dynamic_undo_comparison")
    plot_gs_dcms_comparison(2, 2, 2 * 5 * 272 * 100, 32, "fig_gs_dcms_comparison")
    plot_gs_cms_derivative_comparison(2, 4, 16, "fig_gs_cms_derivative")
    plot_gs_derivative(2, 3, 256, 16, "fig_gs_derivative")
    plot_gs_undo_expand(B=2, L=3, max_cycles=4, granularity=128, count_log_memory=24, count_log_are=24,
                        figure_name="fig_gs_undo_expand")
    plot_gs_cms_static_comparison(2, 4, 16, "fig_gs_cms_static_comparison")
    plot_gs_dcms_granular_comparison(2, 4, 2 * 5 * 272 * 100, 32, "fig_gs_dcms_granular_comparison")
    plot_gs_skew_branching_factor([2, 4, 8, 12, 16], 16, "fig_gs_skew_branching_factor")
    plot_dcms_memory_usage([2, 5], [1000, 500], 32, "fig_dcms_memory_usage")
    plot_branching_factor([2, 4, 8], 16, "fig_branching_factor")

    plot_ip_distribution_zipf("fig_ip_distribution_zipf")
    plot_ip_distribution("fig_ip_distribution")


if __name__ == "__main__":
    parallel()
    plot_dcms_update_query_throughput(8, 6, "fig_dcms_update_query_throughput")
    plot_cms_update_query_throughput(6, 6, 5, 2, "fig_cms_throughput")
    plot_gs_update_query_throughput(8, 6, "fig_gs_update_query_throughput")
    plot_gs_dcms_undo_throughput(8, 6, "fig_gs_dcms_undo_throughput")
    plot_gs_compress_throughput(8, 6, "fig_gs_compress_throughput")