import os
import threading

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
np_noise_array = []


def print_all_raw_data(fast5_file):
    global np_array
    global np_noise_array
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

                for i in range(8):
                    start_idx = randrange(rand_seed)
                    read_segment = raw_data[start_idx:start_idx + 500]
                    read_segment_converted = convert_to_pico(read_segment, fh5.channel_meta['offset'])
                    np_array_temp = np.asarray(read_segment_converted, dtype=np.float32)
                    np_array_temp = np.append(np_array_temp, 1)
                    np_array = np.append(np_array, np_array_temp, axis=0)

                # print_line = ','.join(map(str, read_segment_converted)) + ',0\n'
                # print(print_line)
                # raw_reads.write(print_line)

    if fast5_file_count == 250:
        fast5_file_count = 0
        np_file_count = np_file_count + 1
        np_array = np_array.reshape(-1, 501)
        mu = np.mean(np_noise_array)
        stdev = np.std(np_noise_array)
        np_noise_array = np.random.normal(mu, stdev, np_noise_array.shape)
        np_noise_array = np_noise_array.reshape(-1, 500)
        np_label_array = np.zeros(len(np_noise_array))
        np_noise_array = np.hstack((np_noise_array, np.atleast_2d(np_label_array).T))
        np_array = np.append(np_array, np_noise_array, axis=0)
        np.random.shuffle(np_array)
        print(np_array.shape)
        np.save(TRAIN_DIR + str(np_file_count), np_array)
        np_array = []
        np_noise_array = []


def line_that_contain(string, fp):
    for line in fp:
        if string in line:
            return line


def convert_to_pico(raw_data_arr, offset):
    arr = np.zeros(raw_data_arr.shape, dtype=np.float32)
    for index in range(len(raw_data_arr)):
        arr[index] = (raw_data_arr[index] + offset) * (RANGE / DIGITIZATION)
    return arr


def iterate_dirs(directs):
    # print(directs)
    direc_count = 0
    for direc in directs:
        for fast5_files in os.walk(DATASET + direc):
            if len(fast5_files) > 0:
                direc_count = direc_count + 1
                fast5_count = 0
                for fast5_file in fast5_files:
                    if len(fast5_file) > 0:
                        for k, fi in enumerate(fast5_file):
                            if fi.endswith(".fast5"):
                                fast5_count = fast5_count + 1
                                print(str(fast5_count) + '/' + str(len(fast5_file)) , str(direc_count) + '/' + str(len(directs)))
                                print_all_raw_data(DATASET + '/' + direc + '/' + fi)
                                # if files_count == 5:
                                #    sys.exit()


if __name__ == '__main__':
    raw_reads = open(OUTPUT_FILE, "a")
    nanopore_model = load_model_values(NANOPORE_MODEL)
    reference = ref_reads(REF_FILE)

    files_count = 0
    len_dirs = ''
    files_per = ''
    dirs_list = []
    for root, dirs, files in os.walk(DATASET):
        if len(dirs):
            dirs_list = dirs
    iterate_dirs(dirs_list)
    # t1 = threading.Thread(target=iterate_dirs, args=dirs_list)
    # t2 = threading.Thread(target=iterate_dirs, args=dirs_list[500:1000])
    # t3 = threading.Thread(target=iterate_dirs, args=dirs_list[1000:-1])
    # t1.start()
    # t2.start()
    # t3.start()
    # t1.join()
    # t2.join()
    # t3.join()
    raw_reads.close()
