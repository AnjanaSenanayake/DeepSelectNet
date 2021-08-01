import os
import click
import numpy as np
import matplotlib.pyplot as plt
from fast5_research import Fast5
from ont_fast5_api.fast5_interface import get_fast5_file
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle

colors = ['red', 'blue', 'green', 'yellow']
colors_map = dict()

total_read_count = 0
sc_std = StandardScaler()
sc_robust = RobustScaler()


def get_reads(fast5_file):
    fast5_file_count = 0
    fast5_read_count = 0

    global total_read_count

    with get_fast5_file(fast5_file, mode="r") as f5:
        fast5_file_count = fast5_file_count + 1
        with Fast5(fast5_file) as fh5:
            for read in f5.get_reads():
                # print(fast5_file)
                fast5_read_count = fast5_read_count + 1
                raw_data = read.get_raw_data()

                _range = int(fh5.channel_meta['range'])

                read_converted = convert_to_pico(raw_data,
                                                 fh5.channel_meta['offset'],
                                                 fh5.channel_meta['range'],
                                                 fh5.channel_meta['digitisation'])
                # raw_data = raw_data.reshape(-1, 1)
                # read_normalized = modified_zscore(read_converted, file=fast5_file)

                if _range not in colors_map.keys():
                    colors_map[_range] = colors.pop()

                key = colors_map.get(_range)
                plot_reads(read_converted)
                # boxplot(read_normalized)
                total_read_count = total_read_count + 1


def z_score(data):
    data = data.reshape(-1, 1)
    normalized = sc_std.fit_transform(data)
    return normalized


def robust_scaler(data):
    data = data.reshape(-1, 1)
    normalized = sc_std.fit_transform(data)
    return normalized


def modified_zscore(data, file, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med/(consistency_correction*mad)

    k = np.where(np.abs(mad_score) > 10)
    if len(k[0]) > 0:
        print(file, mad_score[k[0]])

    x = np.where(np.abs(mad_score) > 3)
    x = x[0]

    for i in range(len(x)):

        if x[i] == 0:
            mad_score[x[i]] = mad_score[x[i] + 1]
        elif x[i] == len(mad_score) - 1:
            mad_score[x[i]] = mad_score[x[i] - 1]
        else:
            mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2

    return mad_score


def line_that_contain(string, fp):
    for line in fp:
        if string in line:
            return line


def convert_to_pico(raw_data_arr, _offset, _range, _digitisation):
    arr = np.zeros(raw_data_arr.shape, dtype=np.float32)
    for index in range(len(raw_data_arr)):
        arr[index] = (raw_data_arr[index] + _offset) * (_range / _digitisation)
    return arr


def iterate_dirs_and_read_fast5s(root, directs):
    for direc in directs:
        path = root + direc
        for root, dirs, files in os.walk(path):
            if dirs:
                iterate_dirs_and_read_fast5s(root, dirs)
            if files:
                read_fast5s(root, files)


def read_fast5s(root, files):
    for file in files:
        path = root + '/' + file
        if file.endswith(".fast5"):
            get_reads(path)


def plot_reads(read):
    plt.plot(read, color='blue')
    # plt.pause(0.5)
    # plt.draw()


def boxplot(reads):
    plt.boxplot(reads)
    # plt.pause(0.5)
    # plt.draw()


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory')
def main(fast5_dir):
    for root, dirs, files in os.walk(fast5_dir):
        if dirs:
            iterate_dirs_and_read_fast5s(root, dirs)
        if files:
            read_fast5s(root, files)
    plt.title("Raw value plot for " + str(total_read_count) + " reads")
    # plt.xlabel("ith raw sample of the read")
    plt.ylabel("Normalized raw signal value")
    plt.savefig('figure.png')
    plt.show()


if __name__ == '__main__':
    main()
