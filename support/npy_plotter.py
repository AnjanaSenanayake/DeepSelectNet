import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

MAD_SCORE = 0
REPEATED = False
total_sample_count = 0


def modified_zscore(data, file, MAD, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med/(consistency_correction*mad)

    x = np.where(np.abs(mad_score) > MAD)
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


def plot_npy(path, count, color):
    global total_sample_count
    total_sample_count = 0
    if path.endswith(".npy"):
        data = np.load(path)
        for i, sample in enumerate(data):
            if total_sample_count == count:
                break
            else:
                plt.plot(sample, color=color)
                # plt.boxplot(sample, color=color)
                print("Plotting reads {}".format(total_sample_count))
                total_sample_count += 1


@click.command()
@click.option('--numpy_1', '-npy1', help='Path to npy directory')
@click.option('--numpy_2', '-npy2', help='Path to npy directory')
@click.option('--num_of_reads', '-num', default=1000, help='Number of reads', type=int)
def main(numpy_1, numpy_2, num_of_reads):
    plot_npy(numpy_1, num_of_reads, "blue")
    plot_npy(numpy_2, num_of_reads, "red")
    plt.title("Normalized signal plot for " + str(total_sample_count) + " reads")
    plt.xlabel("ith raw sample of the read")
    plt.ylabel("Normalized raw signal value")
    plt.savefig('figure.png')
    # plt.show()


if __name__ == '__main__':
    main()
