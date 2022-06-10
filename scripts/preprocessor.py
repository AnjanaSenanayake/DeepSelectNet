from timeit import default_timer as timer
import os
import glob
import click
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from random import randrange

FAST5_DIR = ''
CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
SAMPLING_C0 = 1
BATCH = 1000
IS_PICO = True
LABEL = 1
MAD_SCORE = 10
REPEATED = False
OUTPUT = ''
NUM_OF_FILES = 0

file_count = 0
np_dump_count = 0
segment_count = 0


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    x = x[0]

    while True:
        if len(x) > 0:
            for i in range(len(x)):
                if x[i] == 0:
                    mad_score[x[i]] = mad_score[x[i] + 1]
                elif x[i] == len(mad_score) - 1:
                    mad_score[x[i]] = mad_score[x[i] - 1]
                else:
                    mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
        else:
            break

        if REPEATED:
            x = np.where(np.abs(mad_score) > MAD_SCORE)
            x = x[0]
        else:
            break

    return mad_score


def export_numpy(sampled_reads_array):
    global np_dump_count
    np_dump_count += 1
    # print(sampled_reads_array)
    np_array = np.reshape(sampled_reads_array, (-1, SUB_SAMPLE_SIZE + 1))
    np.random.shuffle(np_array)
    np.save(OUTPUT + '/' + str(np_dump_count), np_array)


def print_info():
    if NUM_OF_FILES:
        print("Processing files {}/{}".format(file_count, NUM_OF_FILES))
    else:
        print("No files to preprocess")


def read_fast5s(fast5_dir):
    global file_count
    global segment_count
    sampled_reads_array = []

    s0 = timer()
    for fileNM in glob.glob(fast5_dir + '/*' + ".fast5"):
        file_path = fileNM
        f5 = get_fast5_file(file_path, mode="r")
        file_count += 1

        for read in f5.get_reads():
            raw_data = read.get_raw_data(scale=IS_PICO)

            if (len(raw_data) - CUTOFF) > SUB_SAMPLE_SIZE:
                effective_read = raw_data[CUTOFF:]

                for i in range(SAMPLING_C0):
                    segment_count += 1
                    start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
                    end_idx = start_idx + SUB_SAMPLE_SIZE
                    sampled_read = effective_read[start_idx:end_idx]
                    sampled_read = modified_zscore(sampled_read)
                    sampled_read = np.asarray(sampled_read, dtype=np.float32)
                    sampled_read = np.append(sampled_read, LABEL)
                    sampled_reads_array.append(sampled_read)

                print_info()

        if len(sampled_reads_array) > 0 and (file_count % BATCH == 0):
            # sampled_reads_array = np.asarray(sampled_reads_array, dtype=np.float32)
            export_numpy(sampled_reads_array)
            del sampled_reads_array
            sampled_reads_array = []
    e0 = timer()
    print("Preprocessed in: ", (e0 - s0))


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
@click.option('--cutoff', '-c', default=1500, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=3000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=1, help='subsampling coefficient', type=int)
@click.option('--batch', '-b', default=1000, help='number of fast5 reads for a npy array')
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--label', '-lb', default=1, help='label of the train set', type=int)
@click.option('--mad', '-mad', default=3, help='mad value', type=int)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
@click.option('--output', '-o', help='npy output directory path', type=click.Path(exists=False))
def main(fast5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, label, mad, repeated_norm, output):
    global FAST5_DIR, CUTOFF, SUB_SAMPLE_SIZE, SAMPLING_C0, BATCH, IS_PICO, LABEL, OUTPUT, MAD_SCORE, REPEATED, NUM_OF_FILES
    FAST5_DIR = fast5_dir
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    SAMPLING_C0 = sampling_coefficient
    BATCH = batch
    IS_PICO = pico
    LABEL = label
    MAD_SCORE = mad
    REPEATED = repeated_norm
    OUTPUT = output

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    _, _, files = next(os.walk(FAST5_DIR))
    NUM_OF_FILES = len(files)
    read_fast5s(FAST5_DIR)


if __name__ == '__main__':
    main()
