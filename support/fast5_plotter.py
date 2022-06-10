import os
import click
import numpy as np
import matplotlib.pyplot as plt
from fast5_research import Fast5
from ont_fast5_api.fast5_interface import get_fast5_file
from sklearn.preprocessing import StandardScaler, RobustScaler

MAD_SCORE = 0
REPEATED = False
COUNT = 1000
total_read_count = 0


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    x = x[0]

    while True:
        if len(x) > 0:
            # print(file, mad_score[x[0]])

            for i in range(len(x)):
                if x[i] == 0:
                    mad_score[x[i]] = mad_score[x[i] + 1]
                elif x[i] == len(mad_score) - 1:
                    mad_score[x[i]] = mad_score[x[i] - 1]
                else:
                    mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2

        x = np.where(np.abs(mad_score) > MAD_SCORE)
        x = x[0]
        if ~REPEATED or len(x) <= 0:
            break

    return mad_score


def plot_npy(path, color):
    global total_read_count
    with get_fast5_file(path, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data(scale=True)
            if total_read_count == COUNT:
                break
            else:
                # raw_data = raw_data.reshape(-1, 1)
                normalized_read = modified_zscore(raw_data)

                # plt.plot(normalized_read, color=color)
                plt.boxplot(normalized_read)
                total_read_count = total_read_count + 1
                print("Plotting reads {}".format(total_read_count))


def iterate_dirs_and_read_fast5s(root, directs, color):
    for direc in directs:
        path = root + direc
        for root, dirs, files in os.walk(path):
            if dirs:
                iterate_dirs_and_read_fast5s(root, dirs, color)
            if files:
                read_fast5s(root, files, color)


def read_fast5s(root, files, color):
    for file in files:
        path = root + '/' + file
        if file.endswith(".fast5"):
            plot_npy(path, color)


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory')
@click.option('--mad', '-mad', default=3, help='mad value', type=int)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
@click.option('--num_of_reads', '-num', default=1000, help='Number of reads', type=int)
def main(fast5_dir, mad, repeated_norm, num_of_reads):
    global MAD_SCORE, REPEATED, COUNT
    MAD_SCORE = mad
    REPEATED = repeated_norm
    COUNT = num_of_reads

    for root, dirs, files in os.walk(fast5_dir):
        if dirs:
            iterate_dirs_and_read_fast5s(root, dirs, 'blue')
        if files:
            read_fast5s(root, files, 'blue')

    plt.title("Boxplot for " + str(total_read_count) + " reads")
    # plt.xlabel("n th Signal of the Read")
    plt.ylabel("Normalized Signal Value")
    plt.savefig('boxplot.png')
    plt.show()


if __name__ == '__main__':
    main()
