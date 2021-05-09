import os

from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from LocalConfigs import *

# This can be a single- or multi-read file
# fast5_dir = "datasets/ecoli/fast5_files/"
# fast5_dir = "datasets/covid/single_fast5"
fast5_dir = "datasets/ecoli-covid-sample-pool/pool"

# sample_pool_dir = "datasets/ecoli-covid-sample-pool/ground_negative.txt"
sample_pool_dir = "datasets/ecoli-covid-sample-pool/ground_positive.txt"

raw_reads = open(OUTPUT_FILE, "a")

for root, dirs, files in os.walk(fast5_dir):
    for file in files:
        if file.endswith(".fast5"):
            with get_fast5_file(root + '/' + file, mode="r") as f5:
                for read in f5.get_reads():
                    raw_data = read.get_raw_data()
                    with open(sample_pool_dir, "a") as read_file:
                        read_file.write(read.read_id + '\n')
                        print(read.read_id)

read_file.close()


# arr = np.load(TRAIN_DIR + '/1.npy', allow_pickle=True)
# print(arr)
# print(arr.shape)
