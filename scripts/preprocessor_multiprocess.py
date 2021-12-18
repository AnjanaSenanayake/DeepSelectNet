import math
from multiprocessing import Process, Value, Semaphore
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
OUTPUT = ''

file_count = Value('i', 0)
np_dump_count = Value('i', 0)
segment_count = Value('i', 0)
semaphore = Semaphore()


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


def export_numpy(reads_array, reads_array_temp, output):
    semaphore.acquire()
    np_dump_count.value += 1
    semaphore.release()
    reads_noised_array = add_gaussian_noise(reads_array_temp)
    reads_noised_array = reads_noised_array.reshape(-1, SUB_SAMPLE_SIZE)
    reads_noised_array = np.concatenate(
        [reads_noised_array, np.zeros((reads_noised_array.shape[0], 1), dtype=np.float32)], axis=1)
    reads_noised_array = reads_noised_array.reshape(-1)
    np_array = np.append(reads_array, reads_noised_array, axis=0)
    np_array = np_array.reshape(-1, SUB_SAMPLE_SIZE + 1)
    np.random.shuffle(np_array)
    np.save(output + '/' + str(np_dump_count.value), np_array)


def export_segments(segments_list, output):
    for segment in segments_list:
        semaphore.acquire()
        segment_count.value += 1
        semaphore.release()

        np.save(output + '/' + str(segment_count.value), segment)


def print_info():
    if FILES:
        print("Processing files {}/{}".format(file_count.value, FILES))
    else:
        print("No files to preprocess")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_fast5s(root, files, output):
    reads_array = []
    reads_array_temp = []
    for fast5_file in files:
        file_path = root + '/' + fast5_file
        if fast5_file.endswith(".fast5"):
            with get_fast5_file(file_path, mode="r") as f5:

                semaphore.acquire()
                file_count.value += 1
                semaphore.release()

                with Fast5(file_path) as fh5:
                    for read in f5.get_reads():
                        raw_data = read.get_raw_data()

                        if IS_PICO:
                            raw_data = convert_to_pico(raw_data, fh5)

                        if (len(raw_data) - CUTOFF) >= SUB_SAMPLE_SIZE:
                            effective_read = raw_data[CUTOFF:]

                            for i in range(SAMPLING_C0):
                                semaphore.acquire()
                                segment_count.value += 1
                                semaphore.release()
                                start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
                                end_idx = start_idx + SUB_SAMPLE_SIZE
                                read_sample = effective_read[start_idx:end_idx]
                                read_sample_array = np.asarray(read_sample, dtype=np.float32)
                                reads_array_temp = np.append(reads_array_temp, read_sample_array, axis=0)
                                read_sample_array_temp = np.append(read_sample_array, 1)
                                reads_array = np.append(reads_array, read_sample_array_temp, axis=0)

                if 2*segment_count.value % BATCH == 0:
                    export_numpy(reads_array, reads_array_temp, output)
                    reads_array = []
                    reads_array_temp = []
        print_info()
    if 2*segment_count.value % BATCH != 0:
        export_numpy(reads_array, reads_array_temp, output)


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
@click.option('--cutoff', '-c', default=1000, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=1000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=2, help='subsampling coefficient', type=int)
@click.option('--batch', '-b', default=1000, help='number of fast5 reads for a npy array')
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--output', '-o', help='npy output directory path', type=click.Path(exists=False))
def main(fast5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, output):
    global FAST5_DIR, CUTOFF, SUB_SAMPLE_SIZE, SAMPLING_C0, BATCH, IS_PICO, OUTPUT, DIRS, FILES
    FAST5_DIR = fast5_dir
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    SAMPLING_C0 = sampling_coefficient
    BATCH = batch
    IS_PICO = pico
    OUTPUT = output

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    root, dirs, files = next(os.walk(FAST5_DIR))

    DIRS = len(dirs)
    FILES = len(files)

    files_list = list(chunks(files, 100000))

    processes = []
    for i, sub_list in enumerate(files_list):
        sub = list(sub_list)
        process = Process(target=read_fast5s, args=(root, sub, output))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
