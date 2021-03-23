from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from LocalConfigs import *

# This can be a single- or multi-read file
fast5_filepath = "/storage/e14317/zymo/Zymo-GridION-EVEN-BB-SN/GA10000/reads/105" \
                 "/GXB01153_20181011_FAH71616_GA10000_sequencing_run_EVEN_38808_read_9716_ch_467_strand.fast5"  #

# with get_fast5_file(fast5_filepath, mode="r") as f5:
# 	for read in f5.get_reads():
#         	raw_data = read.get_raw_data()
# 	        print(read.read_id, raw_data)

arr = np.load(TRAIN_DIR + '/1.npy', allow_pickle=True)
print(arr)
print(arr.shape)
