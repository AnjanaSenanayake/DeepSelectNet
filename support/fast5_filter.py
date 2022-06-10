import os
import shutil
import click
from ont_fast5_api.fast5_interface import get_fast5_file

FAST5_DIR = ''
FILTER_TYPE = None
FILTER_OPTIONS = None
CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
READ_IDS_FILE = ''
OUTPUT_DIR = ''

file_count = 0
np_dump_count = 0
segment_count = 0


def read_fast5s(root, files):
    global file_count
    global segment_count
    filtered_count = 0

    if FILTER_OPTIONS == 'length':
        for fast5_file in files:
            file_path = root + '/' + fast5_file
            if fast5_file.endswith(".fast5"):
                file_count += 1
                try:
                    with get_fast5_file(file_path, mode="r") as f5:
                        for read in f5.get_reads():
                            raw_data = read.get_raw_data()
                            if (len(raw_data) - CUTOFF) <= SUB_SAMPLE_SIZE:
                                if FILTER_TYPE == 'move':
                                    shutil.move(file_path, OUTPUT_DIR)
                                    print(file_path, "is moved")
                                elif FILTER_TYPE == 'del':
                                    os.remove(file_path)
                                    print(file_path, "is removed")
                                filtered_count += 1
                except:
                    os.remove(file_path)
                    print(file_path, "is removed")
                print("{}/{} filtered: {} remaining: {}".format(file_count, len(files), filtered_count, (len(files)-filtered_count)))
    elif FILTER_OPTIONS == 'list':
        with open(READ_IDS_FILE, 'rb') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line = line.strip().decode('utf-8') + '.fast5'
                line = str(line)
                file_path = root + '/' + line
                if line in files:
                    if FILTER_TYPE == 'move':
                        if not os.path.exists(OUTPUT_DIR + '/' + line):
                            shutil.move(file_path, OUTPUT_DIR)
                            print(line, "is moved")
                    elif FILTER_TYPE == 'del':
                        os.remove(file_path)
                        print(file_path, "is removed")
                    filtered_count += 1
                print("{}/{} filtered: {} remaining: {}".format(i, len(lines), filtered_count, len(files) - filtered_count))


@click.command()
@click.option('--fast5_dir', '-f5', type=click.Path(exists=True), help='Path to fast5 directory')
@click.option('--f_type', '-t', required=True, type=click.Choice(['move', 'del']),
              help='Filtering types (move--output dir required, delete)')
@click.option('--options', '-opt', required=True, type=click.Choice(['length', 'list']),
              help='Filter on reads length (cutoff length and sample size required)'
                   '| from list of reads-ids (read_ids list required)')
@click.option('--cutoff', '-c', type=int, default=1500, help='Read signal cutoff value (default=1500)')
@click.option('--sample_size', '-sz', type=int, default=3000, help='Read signal sample size (default=3000)')
@click.option('--read_ids', '-ids', type=click.Path(exists=True), help='Read ids list in a file')
@click.option('--output_dir', '-o', type=click.Path(exists=False), help='Path to output directory')
def main(fast5_dir, f_type, options, cutoff, sample_size, read_ids, output_dir):
    global FAST5_DIR, FILTER_TYPE, FILTER_OPTIONS, CUTOFF, SUB_SAMPLE_SIZE, READ_IDS_FILE, OUTPUT_DIR
    FAST5_DIR = fast5_dir
    FILTER_TYPE = f_type
    FILTER_OPTIONS = options
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = sample_size
    READ_IDS_FILE = read_ids
    OUTPUT_DIR = output_dir

    if FILTER_TYPE == 'move' and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    root, dirs, files = next(os.walk(FAST5_DIR))
    read_fast5s(root, files)


if __name__ == '__main__':
    main()
