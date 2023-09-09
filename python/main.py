import math
import os.path

import matplotlib
from scipy import signal
import numpy as np
from scipy.signal import savgol_filter
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# data_folder = "C:\\Users\\USER2\\source\\repos\\DynamicSketch\\DynamicSketch"
data_folder = ""


def plot_error_slope():
    file = open(os.path.join(data_folder, "error_slope.dat"))

    markers = ["o", "D", "s", "^"]
    num_points = 20

    prev_sketch_count = 1

    mean_average_errors = []
    packet_counts = []
    expand_points = []

    mean_average_errors.append([])
    line = file.readline()
    while line:
        if len(line.rstrip()) == 0:
            break
        words = line.rstrip().split()
        sketch_count = int(words[0])
        packet_count = int(words[1])
        mean_average_error = float(words[2])
        if prev_sketch_count != sketch_count:
            prev_sketch_count = sketch_count
            expand_points.append(packet_count)
        mean_average_errors[0].append(mean_average_error)
        packet_counts.append(packet_count)
        line = file.readline()

    while line:
        if len(mean_average_errors[-1]) > 0:
            mean_average_errors.append([])
        while line:
            if len(line.rstrip()) == 0:
                line = file.readline()
                break
            word = line.rstrip()
            mean_average_errors[-1].append(float(word))
            line = file.readline()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    for i in reversed(range(len(mean_average_errors))):
        x = np.array(packet_counts)
        y = np.array(mean_average_errors[i])
        _x = x[::len(packet_counts) // num_points]
        _y = y[::len(packet_counts) // num_points]
        axs[0].plot(_x, _y, marker=markers[i])
        dy = (np.diff(y) / np.diff(x))
        dx = ((np.array(x)[:-1] + np.array(x)[1:]) / 2)
        _y = dy[::len(packet_counts) // num_points]
        _x = dx[::len(packet_counts) // num_points]
        # axs[0][1].plot(_x, _y, marker=markers[i])
        index_expand_first = np.searchsorted(dx, expand_points[0]) + 1
        _y = dy[index_expand_first // 2::len(packet_counts) // num_points]
        _x = dx[index_expand_first // 2::len(packet_counts) // num_points]
        axs[1].plot(_x, _y, marker=markers[i])

    for i, expand_point in enumerate(expand_points):
        color = 'b'
        if i >= len(expand_points) / 2:
            color = 'r'
        axs[0].axvline(x=expand_point, color=color, linestyle='dashed')
        axs[1].axvline(x=expand_point, color=color, linestyle='dashed')

    for ax in axs:
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    axs[0].set_ylabel('MAE')
    axs[0].set_xlabel('UPDATES')
    axs[1].set_ylabel("dMAE/dUPDATES")
    axs[1].set_xlabel('Updates')

    labels = ["count-min sketch " + str(N + 1) for N in reversed(range(len(mean_average_errors) - 1))]
    labels.append("elastic sketch")
    fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=4, fancybox=True, shadow=True)

    fig.savefig('error slope', bbox_inches='tight')
    plt.show()


def plot_operations_per_second():
    file = open(os.path.join(data_folder, "runtime_independent.dat"))

    line = file.readline()
    max_count_sketches = int(line)

    v_query = {}
    v_update = {}

    while line:
        line = file.readline()
        if len(line) == 0:
            break
        words = line.rstrip().split()
        if words[1] == "U":
            x = int(words[0])
            y = float(words[2])
            if x in v_update:
                v_update[x].append(y)
            else:
                v_update[x] = [y]
        elif words[1] == "Q":
            x = int(words[0])
            y = float(words[2])
            if x in v_query:
                v_query[x].append(y)
            else:
                v_query[x] = [y]
        else:
            raise RuntimeError()

    x_query = []
    x_update = []
    y_query = []
    y_update = []
    e_query = []
    e_update = []

    for x in v_query:
        ys = np.array(v_query[x])
        y = ys.mean()
        e = ys.std()
        x_query.append(x)
        y_query.append(y)
        e_query.append(e)

    for x in v_update:
        ys = np.array(v_update[x])
        y = ys.mean()
        e = ys.std()
        x_update.append(x)
        y_update.append(y)
        e_update.append(e)

    number_points = 16
    d = len(x_query) // number_points

    x_query = x_query[::d]
    x_update = x_update[::d]
    y_update = np.array(y_update[::d])
    y_query = np.array(y_query[::d])
    e_update = np.array(e_update[::d])
    e_query = np.array(e_query[::d])

    plt.plot(x_update, y_update, marker="o", label="update")

    plt.plot(x_query, y_query, marker="D", label="query")

    plt.ylabel('Operations/second')
    plt.xlabel('Dynamic sketch size')
    plt.legend()
    plt.grid()
    plt.show()


def plot_operations_table():
    file = open(os.path.join(data_folder, "latency.dat"))

    line = file.readline().rstrip()
    max_count_sketch, aggregate_size = [int(word) for word in line.split()]

    tree_update_times = []
    tree_query_times = []
    hash_update_times = []
    hash_query_times = []
    sketch_times = [{"update": [], "query": [], "expand": [], "shrink": []} for i in range(max_count_sketch)]

    while line:
        line = file.readline().rstrip()
        if len(line) == 0:
            break
        words = line.rstrip().split()
        sketch_count, tree_update_time, tree_query_time, _, hash_update_time, hash_query_time, _, dynamic_update_time, \
        dynamic_query_time, _, phase, modify_size_time = words
        tree_query_times.append(int(tree_query_time) / aggregate_size)
        tree_update_times.append(int(tree_update_time) / aggregate_size)
        hash_query_times.append(int(hash_query_time) / aggregate_size)
        hash_update_times.append(int(hash_update_time) / aggregate_size)
        sketch_times[int(sketch_count) - 1]["update"].append(int(dynamic_update_time))
        sketch_times[int(sketch_count) - 1]["query"].append(int(dynamic_query_time))
        if phase == "E":
            sketch_times[int(sketch_count) - 1]["expand"].append(int(modify_size_time))
        elif phase == "S":
            sketch_times[int(sketch_count) - 1]["shrink"].append(int(modify_size_time))
        else:
            raise RuntimeError("phase error")

    tree_update_times = np.array(tree_update_times)
    tree_query_times = np.array(tree_query_times)
    hash_update_times = np.array(hash_update_times)
    hash_query_times = np.array(hash_query_times)

    sizes = [4, 8, 16, 32]
    columns = ('Update P50/P90/P99', 'Query P50/P90/P99', 'Expand P50/P90/P99', 'Shrink P50/P90/P99')
    rows = ['Red-black tree', 'Hashtable']
    rows += ["Dynamic sketch of size " + str(i) for i in sizes]

    data = []
    for update, query in [[tree_update_times, tree_query_times], [hash_update_times, hash_query_times]]:
        data.append([
            '/'.join([str(round(np.percentile(update, p))) for p in [50, 90, 99]]),
            '/'.join([str(round(np.percentile(query, p))) for p in [50, 90, 99]]),
            "X/X/X",
            "X/X/X"
        ])

    for s in sizes:
        update = np.array(sketch_times[s - 1]["update"])
        query = np.array(sketch_times[s - 1]["query"])
        expand = np.array(sketch_times[s - 1]["expand"])
        shrink = np.array(sketch_times[s - 1]["shrink"])
        dataline = []
        for ts in [update, query, expand, shrink]:
            dataline.append('/'.join([str(round(np.percentile(ts, p))) for p in [50, 90, 99]]))
        data.append(dataline)

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellLoc='center',
             cellText=data,
             rowLabels=rows,
             colLabels=columns,
             loc='center')

    fig.tight_layout()

    plt.show()


def plot_runtime():
    file = open(os.path.join(data_folder, "runtime.dat"))

    runtimes = {"tree": [], "hash": []}
    for line in file.readlines():
        words = line.strip().split(" ")
        if len(words) == 2:
            runtimes["tree"].append(float(words[0]))
            runtimes["hash"].append(float(words[1]))
        else:
            key = "f" + words[0] + "a" + words[1]
            if key not in runtimes:
                runtimes[key] = []
            runtimes[key].append(float(words[2]))

    e = []
    runtime_keys = runtimes.keys()
    x_pos = np.arange(len(runtime_keys))
    CTEs = [np.array(l).mean() for l in runtimes.values()]
    std = [np.array(l).std() for l in runtimes.values()]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Seconds')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(runtime_keys)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('runtime trace.png')
    plt.show()


def plot_modify_size():
    file = open(os.path.join(data_folder, "modify_size.dat"))

    data = []
    labels = []

    i = 0
    for line in file.readlines():
        words = line.strip().split(" ")
        if len(line) == 0:
            break
        if i % 2 == 0:
            labels.append((int(words[0]), int(words[1])))
        else:
            data.append([float(word) for i, word in enumerate(words) if i % 2 == 0])
            data.append([float(word) for i, word in enumerate(words) if i % 2 == 1])
        i += 1

    nrows = int(math.sqrt(len(labels))) // 2
    fig, axs = plt.subplots(nrows=nrows, ncols=2)
    freqs = list({f for (f, _) in labels})
    ampls = list({a for (_, a) in labels})
    markers = ["o", "D", "s", "^"]
    n_points = 10

    for j, a in enumerate(ampls):
        ampl_data = []
        for i, l in enumerate(labels):
            if l[1] == a:
                x = data[i * 2]
                y = data[i * 2 + 1]
                x = x[::len(x) // n_points]
                y = y[::len(y) // n_points]
                axs[j // 2][j % nrows].plot(x, y, marker=markers[freqs.index(l[0])],
                                            label="frequency " + str(labels[i][0]))
                axs[j // 2][j % nrows].set_xlabel('packet count')
                axs[j // 2][j % nrows].set_ylabel('MAE')
                axs[j // 2][j % nrows].set_title("amplitude " + str(a))
                axs[j // 2][j % nrows].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    fig.legend(["amplitude " + str(f) for f in freqs], loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=2, fancybox=True, shadow=True)
    plt.tight_layout()
    fig.savefig('modify_size', bbox_inches='tight')
    plt.show()

def plot_naive():
    file = open(os.path.join(data_folder, "naive.dat"))

    dynamic_mean_average_errors = []
    naive_mean_average_errors = []
    expand_counts = []

    while True:
        line = file.readline()
        words = line.rstrip().split()
        if not line or len(words) < 2:
            break
        dynamic_mean_average_errors.append(float(words[1]))
        line = file.readline()
        words = line.rstrip().split()
        naive_mean_average_errors.append(float(words[1]))
        expand_counts.append(int(words[0]))
    fig, axs = plt.subplots(2)

    markers = ["o", "D"]
    for i, maes in enumerate([dynamic_mean_average_errors, naive_mean_average_errors]):
        x = np.array(expand_counts)
        y = np.array(maes)
        axs[0].plot(x, y,  marker=markers[i])
        y = np.diff(y) / np.diff(x)
        x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
        axs[1].plot(x, y, marker=markers[i])

    axs[0].set_ylabel('MAE')
    axs[0].set_xlabel('Sketch size')
    axs[1].set_ylabel('dMAE/dSIZE')
    axs[1].set_xlabel('Sketch size')

    plt.legend()
    plt.show()

def plot_expand_improvement_bound():
    file = open(os.path.join(data_folder, "expand_improvement_bound.dat"))

    mean_average_errors = []
    expand_counts = []

    for line in file.readlines():
        words = line.rstrip().split()
        if len(words) < 2:
            break
        mean_average_errors.append(float(words[1]))
        expand_counts.append(int(words[0]))

    fig, axs = plt.subplots(2)

    x = np.array(expand_counts)
    y = np.array(mean_average_errors)
    axs[0].plot(x, y)
    y = np.diff(y) / np.diff(x)
    x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
    axs[1].plot(x, y)

    axs[0].set_ylabel('mae')
    axs[0].set_xlabel('expand_count')
    axs[1].set_ylabel('d_mae/d_expand_count')
    axs[1].set_xlabel('expand_count')

    axs[0].set_title("MAE and dMAE in dynamic sketch during a single run, expands equally distanced between updates")

    plt.show()


def plot_memory_usage():
    file = open(os.path.join(data_folder, "memory_usage.dat"))

    data = {"dynamic sketch 1 size": [], "dynamic sketch 2 size": [],
            "dynamic sketch 1 mse ": [], "dynamic sketch 2 mse ": [], "binary tree size": [],
            "hash table size": []}
    packet_count = []

    for line in file.readlines():
        words = [float(word) for word in line.strip().split(" ")]
        data["hash table size"].append(words[0])
        data["binary tree size"].append(words[1])
        data["dynamic sketch 1 size"].append(words[2])
        data["dynamic sketch 2 size"].append(words[3])
        data["dynamic sketch 1 mse "].append(words[4])
        data["dynamic sketch 2 mse "].append(words[5])
        packet_count.append(words[6])
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    for k in data.keys():
        ax = axs[0]
        if k.endswith("size"):
            ax = axs[1]
        ax.plot(packet_count, data[k], label=k[:len(k) - 5])

    axs[0].set_xlabel('packet count')
    axs[0].set_ylabel('mse')
    axs[1].set_xlabel('packet count')
    axs[1].set_ylabel('size in bytes')

    axs[0].grid()
    axs[1].grid()

    axs[0].legend()
    axs[1].legend()

    plt.show()


def plot_expand_accuracy():
    file = open(os.path.join(data_folder, "expand_accuracy.dat"))

    data = []
    labels = ['count min sized 1']

    for line in file.readlines():
        words = line.strip().split(" ")
        if len(line) == 0:
            break
        if len(labels) == 1:
            labels.append('count min sized ' + str(words[0]))
            labels += ['dynamic sketch with ' + str(i) + ' expands' for i in words[1:]]
            continue
        data.append([float(word) for i, word in enumerate(words) if i % 2 == 0])
        data.append([float(word) for i, word in enumerate(words) if i % 2 == 1])

    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(0, len(data) // 2):
        ax.plot(data[i * 2], data[i * 2 + 1], label=labels[i])

    ax.set_title('MSE of different sketchs')
    ax.set_xlabel('packet count')
    ax.set_ylabel('mean square error')
    ax.legend()
    plt.show()


def plot_amplitude_frequency_explanation():
    fig, ax = plt.subplots(1)

    t = np.linspace(0, 1, 1000)
    triangle = signal.sawtooth(np.pi * 4 * t, 0.5)
    ax.plot(t, triangle * 29 + 31)
    ax.set_xlabel('Updates percentage')
    ax.set_ylabel('Dynamic sketch size')

    plt.grid()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5, 4, forward=True)
    fig.savefig('terminology', bbox_inches='tight')

    plt.show()


    # ax.set_yticklabels([])
    # ax.set_xticklabels([])


plot_naive()

# accuracy
plot_modify_size()  # amplitude / frequency
plot_error_slope()

# memory
plot_memory_usage()

# runtime
plot_operations_table()  # table
plot_operations_per_second()  # query & update
# plot_runtime()  # runtime of entire trace


# plot_expand_improvement_bound()
# plot_expand_accuracy()
