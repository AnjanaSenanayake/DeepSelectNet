import os
import click
from ont_fast5_api.fast5_interface import get_fast5_file

total_read_count = 0
total_read_length = 0


def get_reads(fast5_file):
    global total_read_count
    global total_read_length

    with get_fast5_file(fast5_file, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            total_read_count += 1
            total_read_length += len(raw_data)
            print("Read Id: {} Read Count: {}, Read Length:{} Total Read Length: {}".format(read.read_id, total_read_count, len(raw_data),total_read_length))


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
