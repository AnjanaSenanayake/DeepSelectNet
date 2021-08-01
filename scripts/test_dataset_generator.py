import os
import click
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from fast5_research import Fast5
from random import randrange

FAST5_DIR = ''
CUTOFF = 1000
SUB_SAMPLE_SIZE = 1000
SAMPLING_C0 = 2
BATCH = 1000
IS_PICO = True
LABEL = 0
DUMP_VALUE = 0
OUTPUT = ''

file_count = 0
np_dump_count = DUMP_VALUE
segment_count = 0


def add_gaussian_noise(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    noise = np.random.normal(mean, std, arr.shape)
    noisy_array = arr + noise
    return noisy_array


def convert_to_pico(raw_read, fh5):
    _range = int(fh5.channel_meta['range'])
    _offset = int(fh5.channel_meta['offset'])
    _digitisation = int(fh5.channel_meta['digitisation'])
    arr = np.zeros(raw_read.shape, dtype=np.float32)
    for index in range(len(raw_read)):
        arr[index] = (raw_read[index] + _offset) * (_range / _digitisation)
    return arr


def export_numpy(reads_array):
    global np_dump_count
    np_dump_count += 1
    np_array = reads_array.reshape(-1, SUB_SAMPLE_SIZE + 1)
    np.random.shuffle(np_array)
    np.save(OUTPUT + '/' + str(np_dump_count), np_array)


def print_info(num_of_files):
    if num_of_files:
        print("Processing files {}/{}".format(file_count, num_of_files))
    else:
        print("No files to preprocess")


def read_fast5s(root, files):
    global file_count
    global segment_count
    reads_array = []
    for fast5_file in files:
        file_path = root + '/' + fast5_file
        if fast5_file.endswith(".fast5"):
            with get_fast5_file(file_path, mode="r") as f5:
                file_count += 1

                with Fast5(file_path) as fh5:
                    for read in f5.get_reads():
                        raw_data = read.get_raw_data()

                        if IS_PICO:
                            raw_data = convert_to_pico(raw_data, fh5)

                        if (len(raw_data) - CUTOFF) >= SUB_SAMPLE_SIZE:
                            effective_read = raw_data[CUTOFF:]

                            for i in range(SAMPLING_C0):
                                segment_count += 1
                                start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
                                end_idx = start_idx + SUB_SAMPLE_SIZE
                                read_sample = effective_read[start_idx:end_idx]
                                read_sample_array = np.asarray(read_sample, dtype=np.float32)
                                read_sample_array_temp = np.append(read_sample_array, LABEL)
                                reads_array = np.append(reads_array, read_sample_array_temp, axis=0)

                if segment_count % BATCH == 0:
                    export_numpy(reads_array)
                    reads_array = []
        print_info(len(files))

    if segment_count % BATCH != 0:
        export_numpy(reads_array)


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
@click.option('--cutoff', '-c', default=1000, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=1000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=2, help='subsampling coefficient', type=int)
@click.option('--batch', '-b', default=1000, help='number of fast5 reads for a npy array')
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--label', '-lbl', default=0, help='class value', type=int)
@click.option('--np_dump_start_value', '-dump_start', default=0, help='starting value for np dump', type=int)
@click.option('--output', '-o', help='npy output directory path', type=click.Path(exists=False))
def main(fast5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, label, np_dump_start_value, output):
    global FAST5_DIR, CUTOFF, SUB_SAMPLE_SIZE, SAMPLING_C0, BATCH, IS_PICO, LABEL, DUMP_VALUE, OUTPUT
    FAST5_DIR = fast5_dir
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    SAMPLING_C0 = sampling_coefficient
    BATCH = batch
    IS_PICO = pico
    LABEL = label
    DUMP_VALUE = np_dump_start_value
    OUTPUT = output

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    root, dirs, files = next(os.walk(FAST5_DIR))
    read_fast5s(root, files)


if __name__ == '__main__':
    main()
