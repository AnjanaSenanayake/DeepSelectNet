import os
import sys
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from main import load_model_values, ref_reads, call_raw_values
from fast5_research import Fast5

NANOPORE_MODEL = "nanopore_model.txt"
REF_FILE = "../datasets/covid/nCov-2019-ref.fasta"
DATASET = "../datasets/covid/SP1-raw/single_reads/"
OUTPUT_FILE = "../datasets/covid/nn_data.txt"
BASE_SAMPLING_RATE = 12
DIGITIZATION = 8192
RANGE = 1402.882


def print_all_raw_data(fast5_file):
    with get_fast5_file(fast5_file, mode="r") as f5:
        with Fast5(fast5_file) as fh5:
            for read in f5.get_reads():
                with open('../datasets/covid/out.paf') as paf:
                    raw_data = read.get_raw_data()
                    # print(read.read_id, raw_data)
                    paf_line = line_that_contain(read.read_id, paf)
                    paf_line_tokens = paf_line.split()

                    # Basic Raw Data Mapping
                    # print(raw_data, paf_line_tokens[7], paf_line_tokens[8])
                    # print(raw_data)

                    # Base level sampling
                    # print(len(raw_data), int(paf_line_tokens[1]), len(raw_data) / int(paf_line_tokens[1]))

                    # raw_data_strt_idx = BASE_SAMPLING_RATE * int(paf_line_tokens[2])
                    # raw_data_end_idx = BASE_SAMPLING_RATE * int(paf_line_tokens[3])
                    # print(convert_to_pico(np.asarray(raw_data, dtype=np.float32), fh5.channel_meta['offset']))

                    np.set_printoptions(threshold=sys.maxsize)
                    raw_data_arr = np.asarray(raw_data[0:4000], dtype=np.float32)
                    raw_data_arr = convert_to_pico(raw_data_arr, fh5.channel_meta['offset'])
                    raw_data_arr = ' '.join(map(str, raw_data_arr))

                    # Raw signal size vs ref mapping size
                    # print(len(raw_data), len(raw_data_arr))

                    ref_map_strt_idx = int(paf_line_tokens[7])
                    ref_map_end_idx = int(paf_line_tokens[8])
                    # ref_raw_signal = call_raw_values(reference[ref_map_strt_idx:ref_map_end_idx + 1], nanopore_model)
                    # ref_data_arr = np.asarray(ref_raw_signal, dtype=np.float32)

                    # ref_data_arr = ' '.join(map(str, ref_data_arr))
                    # # actual_query_sequence_length = int(paf_line_tokens[3]) - int(paf_line_tokens[2])
                    # # print(actual_query_sequence_length, len(ref_raw_signal), '\n')
                    # print_line = str(raw_data_arr) + ', ' + str(ref_data_arr) + '\n'
                    # # print(print_line)
                    # raw_reads.write(print_line)


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
                if files_count == 11:
                    break
    raw_reads.close()
