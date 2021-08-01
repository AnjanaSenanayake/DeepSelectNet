import shutil
import sys

import click
import matplotlib.pyplot as plt

FAST5_DIR = ''
FAST5_LIST = ''
OUTPUT = ''
FILES_COUNT = 0


@click.command()
@click.option('--tsv_file', '-tsv', help='path to tsv file', type=click.Path(exists=True))
@click.option('--batch', '-b', help='tsv batch size', type=int)
def main(tsv_file, batch):
    last_read_id = ''
    min_signal_idx = float('inf')

    count = 0
    batches = 0
    event_list = []
    event_sorted_list = []
    for line in open(tsv_file):
        if count == batch:
            batches = batches + 1
            print(last_read_id, min_signal_idx)
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.hist(event_list)
            ax2.hist(event_sorted_list)
            fig.savefig('event_sorts-' + str(batches) + '.png')

            print(str(count) + "/" + str(batches))
            count = 0
            event_list = []
            event_sorted_list = []
        read_data = line.strip().split('\t')
        if read_data[0] == 'contig':
            pass
        else:
            read_id = read_data[3]
            signal_idx = int(read_data[13])
            event_list.append(signal_idx)
            count = count + 1

            if last_read_id == read_id:
                if signal_idx < min_signal_idx:
                    min_signal_idx = signal_idx
            else:
                # print(last_read_id, min_signal_idx)
                if min_signal_idx != float('inf'):
                    event_sorted_list.append(min_signal_idx)
                last_read_id = read_id
                min_signal_idx = signal_idx
    # print(last_read_id, min_signal_idx)
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.hist(event_list)
    # ax2.hist(event_sorted_list)
    # fig.savefig('event_sorts.png')


if __name__ == '__main__':
    main()
