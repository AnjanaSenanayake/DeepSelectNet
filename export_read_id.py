import os
import sys
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from LocalConfigs import *

for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if file.endswith(".fast5"):
            with get_fast5_file(root + '/' + file, mode="r") as f5:
                for read in f5.get_reads():
                    raw_data = read.get_raw_data()
                    with open(sys.argv[2], "a") as read_file:
                        read_file.write(read.read_id + '\n')
                        print(read.read_id)
read_file.close()
