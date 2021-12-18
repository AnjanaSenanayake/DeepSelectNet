import os
import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

MAD_SCORE = 0
REPEATED = False
total_sample_count = 0
sc_std = StandardScaler()
sc_robust = RobustScaler()


def get_samples(npy_file):
    global total_sample_count

    data = np.load(npy_file)
    for i, sample in enumerate(data):
        sample = modified_zscore(data=sample, file=i)

        # plot_reads(sample)
        boxplot(sample)
        total_sample_count = total_sample_count + 1


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

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    x = x[0]

    while True:
        if len(x) > 0:
            print(file, mad_score[x[0]])

            for i in range(len(x)):
                if x[i] == 0:
                    mad_score[x[i]] = mad_score[x[i] + 1]
                elif x[i] == len(mad_score) - 1:
                    mad_score[x[i]] = mad_score[x[i] - 1]
                else:
                    mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2

        x = np.where(np.abs(mad_score) > MAD_SCORE)
        x = x[0]
        if REPEATED or len(x) <= 0:
            break

    return mad_score


def convert_to_pico(raw_data_arr, _offset, _range, _digitisation):
    arr = np.zeros(raw_data_arr.shape, dtype=np.float32)
    for index in range(len(raw_data_arr)):
        arr[index] = (raw_data_arr[index] + _offset) * (_range / _digitisation)
    return arr


def read_npys(root, files):
    for file in files:
        path = root + '/' + file
        if file.endswith(".npy"):
            get_samples(path)


def plot_reads(read):
    plt.plot(read, color='blue')
    # plt.pause(0.5)
    # plt.draw()


def boxplot(reads):
    plt.boxplot(reads)
    # plt.pause(0.5)
    # plt.draw()


@click.command()
@click.option('--numpy', '-np', help='path to fast5 directory')
@click.option('--mad', '-mad', default=3, help='mad value', type=int)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
def main(numpy, mad, repeated_norm):
    global MAD_SCORE, REPEATED
    MAD_SCORE = mad
    REPEATED = repeated_norm
    for root, _, files in os.walk(numpy):
        read_npys(root, files)
    plt.title("Raw value plot for " + str(total_sample_count) + " reads")
    plt.xlabel("ith raw sample of the read")
    plt.ylabel("Normalized raw signal value")
    plt.savefig('figure.png')
    plt.show()


if __name__ == '__main__':
    main()
