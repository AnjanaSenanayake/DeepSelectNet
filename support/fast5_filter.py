import os
import click
from ont_fast5_api.fast5_interface import get_fast5_file
from fast5_research import Fast5

FAST5_DIR = ''
CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
FILTER_TYPE = ''
COMPARE_LIST = ''

file_count = 0
np_dump_count = 0
segment_count = 0


def read_fast5s(root, files):
    global file_count
    global segment_count
    filtered_count = 0
    for fast5_file in files:
        file_path = root + '/' + fast5_file
        if fast5_file.endswith(".fast5"):
            with get_fast5_file(file_path, mode="r") as f5:
                file_count += 1
                for read in f5.get_reads():
                    raw_data = read.get_raw_data()

                    if FILTER_TYPE == 'Length':
                        if (len(raw_data) - CUTOFF) <= SUB_SAMPLE_SIZE:
                            os.remove(file_path)
                            print(file_path, "is removed")
                            filtered_count += 1
                    elif FILTER_TYPE == 'List':
                        with open(COMPARE_LIST, 'rb') as file:
                            lines = file.readlines()
                            for line in lines:
                                line = line.strip().decode('utf-8')
                                if read.read_id == line:
                                    os.remove(file_path)
                                    print(file_path, "is removed")
                                    filtered_count += 1
                                    break
                    print("{}/{} filtered files: {}".format(file_count, len(files), filtered_count))


@click.command()
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
@click.option('--cutoff', '-c', default=1500, help='read signal cutoff value(default=1500)', type=int)
@click.option('--subsample_size', '-sz', default=3000, help='read signal sample size(default=3000)', type=int)
@click.option('--filter_type', '-type', default='Length', help='Filter types(options=[Length,List], default=Length)', type=str)
@click.option('--compare_list', '-list', default='', help='Compare list', type=click.Path(exists=False))
def main(fast5_dir, cutoff, subsample_size, filter_type, compare_list):
    global FAST5_DIR, CUTOFF, SUB_SAMPLE_SIZE, FILTER_TYPE, COMPARE_LIST
    FAST5_DIR = fast5_dir
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    FILTER_TYPE = filter_type
    COMPARE_LIST = compare_list

    root, dirs, files = next(os.walk(FAST5_DIR))
    read_fast5s(root, files)


if __name__ == '__main__':
    main()
