import os
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from main import load_model_values, ref_reads, call_raw_values
from fast5_research import Fast5
from random import randrange
from LocalConfigs import *

BASE_SAMPLING_RATE = 12
DIGITIZATION = 8192
RANGE = 1402.882

fast5_file_count = 0
np_file_count = 0
np_array = []


def print_all_raw_data(fast5_file):
    global np_array
    global fast5_file_count
    global np_file_count
    with get_fast5_file(fast5_file, mode="r") as f5:
        fast5_file_count = fast5_file_count + 1
        with Fast5(fast5_file) as fh5:
            for read in f5.get_reads():
                # with open(PAF_FILE) as paf:
                raw_data = read.get_raw_data()
                # print(read.read_id)
                # paf_line = line_that_contain(read.read_id, paf)
                # paf_line_tokens = paf_line.split()

                rand_seed = len(raw_data) - 500

                for i in range(5):
                    start_idx = randrange(rand_seed)
                    read_segment = raw_data[start_idx:start_idx + 500]
                    read_segment_converted = convert_to_pico(read_segment, fh5.channel_meta['offset'])
                    np_array_temp = np.asarray(read_segment_converted, dtype=np.float32)
                    np_array_temp = np.append(np_array_temp, 1)
                    np_array = np.append(np_array, np_array_temp, axis=0)

                    mu = np.mean(read_segment_converted)
                    stdev = np.std(read_segment_converted)
                    noise_array = np.random.normal(mu, stdev, read_segment_converted.shape)
                    noise_array = np.append(noise_array, 0)
                    np_array = np.append(np_array, noise_array, axis=0)

                    # print_line = ','.join(map(str, read_segment_converted)) + ',0\n'
                    # print(print_line)
                    # raw_reads.write(print_line)

    if fast5_file_count == 1000:
        fast5_file_count = 0
        np_file_count = np_file_count + 1
        np_array = np_array.reshape(-1, 501)
        print(np_array.shape)
        np.save(TRAIN_DIR + str(np_file_count), np_array)
        np_array = []


def line_that_contain(string, fp):
    for line in fp:
        if string in line:
            return line


def convert_to_pico(raw_data_arr, offset):
    arr = np.zeros(raw_data_arr.shape, dtype=np.float32)
    for index in range(len(raw_data_arr)):
        arr[index] = (raw_data_arr[index] + offset) * (RANGE / DIGITIZATION)
    return arr


if __name__ == '__main__':
    raw_reads = open(OUTPUT_FILE, "a")
    nanopore_model = load_model_values(NANOPORE_MODEL)
    reference = ref_reads(REF_FILE)

    files_count = 0
    len_dirs = ''
    files_per = ''
    for root, dirs, files in os.walk(DATASET):
        if len(dirs):
            files_count = files_count + 1
            len_dirs = str(len(dirs))
            files_per = str(files_count) + '/' + len_dirs
        file_count = 0
        for file in files:
            if file.endswith(".fast5"):
                file_count = file_count + 1
                file_path = root + '/' + file
                print_all_raw_data(file_path)
                if file_count == len(files):
                    files_count = files_count + 1
                    files_per = str(files_count) + '/' + len_dirs
                print(str(file_count) + '/' + str(len(files)) + ' | ' + files_per)
                if files_count == 5:
                    break
    raw_reads.close()
