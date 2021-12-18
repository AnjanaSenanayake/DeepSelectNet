import os
import click
import numpy as np
from fast5_research import Fast5
from ont_fast5_api.fast5_interface import get_fast5_file

total_read_count = 0


def get_reads(fast5_file):
    fast5_file_count = 0

    global total_read_count

    with get_fast5_file(fast5_file, mode="r") as f5:
        fast5_file_count = fast5_file_count + 1
        with Fast5(fast5_file) as fh5:
            for read in f5.get_reads():
                raw_data = read.get_raw_data()
                print(raw_data, len(raw_data))
                total_read_count = total_read_count + 1


def convert_to_pico(raw_data_arr, _offset, _range, _digitisation):
    arr = np.zeros(raw_data_arr.shape, dtype=np.float32)
    for index in range(len(raw_data_arr)):
        arr[index] = (raw_data_arr[index] + _offset) * (_range / _digitisation)
    return arr


def read_fast5s(root, files):
    for file in files:
        path = root + '/' + file
        if file.endswith(".fast5"):
            get_reads(path)


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
def main(fast5_dir):
    for root, _, files in os.walk(fast5_dir):
        if files:
            read_fast5s(root, files)
    print("Total read counts: ", total_read_count)


if __name__ == '__main__':
    main()
