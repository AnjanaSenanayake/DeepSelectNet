import shutil
import sys
from timeit import default_timer as timer
import pyslow5
import os
import click
import numpy as np
from random import randrange

POS_SLOW5_DIR = ''
NEG_SLOW5_DIR = ''
CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
SAMPLING_C0 = 1
BATCH = 1000
IS_PICO = True
LABEL = 1
MAD_SCORE = 10
REPEATED = False
OUTPUT = ''
NUM_OF_READS = 0

read_count = 0
rejected_count = 0
np_dump_count = 0
segment_count = 0
stats = {'t_pos_reads': 0, 't_neg_read': 0, 'r_pos_reads': 0, 'r_neg_reads': 0}


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    x = x[0]

    while True:
        if len(x) > 0:
            for i in range(len(x)):
                if x[i] == 0:
                    mad_score[x[i]] = mad_score[x[i] + 1]
                elif x[i] == len(mad_score) - 1:
                    mad_score[x[i]] = mad_score[x[i] - 1]
                else:
                    mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
        else:
            break

        if REPEATED:
            x = np.where(np.abs(mad_score) > MAD_SCORE)
            x = x[0]
        else:
            break

    return mad_score


def export_numpy(sampled_reads_array):
    global np_dump_count
    np_dump_count += 1
    dump_name = 'pos'
    np_array = np.reshape(sampled_reads_array, (-1, SUB_SAMPLE_SIZE + 1))
    np.random.shuffle(np_array)
    if LABEL == 0:
        dump_name = 'neg'
    np.save(OUTPUT + '/' + dump_name, np_array)


def print_info():
    if NUM_OF_READS:
        if LABEL == 1:
            print("Processing positive reads {}/{}".format(read_count, NUM_OF_READS))
        else:
            print("Processing negative reads {}/{}".format(read_count, NUM_OF_READS))
    else:
        print("No files to preprocess")


def read_slow5s(slow5_dir):
    global NUM_OF_READS, read_count, rejected_count, segment_count, stats
    read_count = 0
    rejected_count = 0
    sampled_reads_array = []

    s0 = timer()
    s5 = pyslow5.Open(slow5_dir, 'r')
    _, NUM_OF_READS = s5.get_read_ids()
    reads = s5.seq_reads(pA=IS_PICO)

    if NUM_OF_READS < BATCH:
        print("\n-----Preprocessing Failed-----")
        print("Number of reads should be equal or more than batch process size")
        print("Num. of reads = {}, Batch Size = {}".format(NUM_OF_READS, BATCH))
        shutil.rmtree(OUTPUT)
        exit(0)

    for read in reads:
        raw_data = read['signal']

        if (read['len_raw_signal'] - CUTOFF) > SUB_SAMPLE_SIZE:
            effective_read = raw_data[CUTOFF:]

            for i in range(SAMPLING_C0):
                segment_count += 1
                start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
                end_idx = start_idx + SUB_SAMPLE_SIZE
                sampled_read = effective_read[start_idx:end_idx]
                sampled_read = modified_zscore(sampled_read)
                sampled_read = np.asarray(sampled_read, dtype=np.float32)
                sampled_read = np.append(sampled_read, LABEL)
                sampled_reads_array.append(sampled_read)
            read_count += 1
            print_info()
        else:
            rejected_count += 1

        if len(sampled_reads_array) > 0 and (read_count % BATCH == 0):
            export_numpy(sampled_reads_array)
            sampled_reads_array = []
            break
    e0 = timer()
    print("Preprocessed in: {} seconds".format(e0 - s0))


@click.command()
@click.option('--positive_slow5_dir', '-pos_s5', help='path to positive class slow5 directory',
              type=click.Path(exists=True))
@click.option('--negative_slow5_dir', '-neg_s5', help='path to negative class slow5 directory',
              type=click.Path(exists=True))
@click.option('--cutoff', '-c', default=1500, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=3000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=1, help='subsampling coefficient', type=int)
@click.option('--batch', '-b', default=1000, help='number of fast5 reads for a npy array')
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--mad', '-mad', default=3, help='mad value', type=int)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
@click.option('--output', '-o', help='npy output directory path', type=click.Path(exists=False))
def main(positive_slow5_dir, negative_slow5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, mad,
         repeated_norm, output):
    global POS_SLOW5_DIR, NEG_SLOW5_DIR, CUTOFF, SUB_SAMPLE_SIZE, SAMPLING_C0, BATCH, IS_PICO, LABEL, OUTPUT, MAD_SCORE, REPEATED
    POS_SLOW5_DIR = positive_slow5_dir
    NEG_SLOW5_DIR = negative_slow5_dir
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    SAMPLING_C0 = sampling_coefficient
    BATCH = batch
    IS_PICO = pico
    MAD_SCORE = mad
    REPEATED = repeated_norm
    OUTPUT = output

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    LABEL = 1
    read_slow5s(POS_SLOW5_DIR)
    stats['t_pos_reads'] = read_count
    stats['r_pos_reads'] = rejected_count

    LABEL = 0
    read_slow5s(NEG_SLOW5_DIR)
    stats['t_neg_read'] = read_count
    stats['r_neg_reads'] = rejected_count

    print("Total processed positive reads: {}".format(stats['t_pos_reads']-stats['r_pos_reads']))
    print("Total ignored positive reads: {}".format(stats['r_pos_reads']))
    print("Total processed negative reads: {}".format(stats['t_neg_read']-stats['r_neg_reads']))
    print("Total ignored negative reads: {}".format(stats['r_neg_reads']))

    if stats['t_pos_reads'] != stats['t_neg_read']:
        print("\n-----Preprocessing Failed-----")
        print("Num. of processed positive reads and processed negative reads are in-equal. Dump is imbalanced", file=sys.stderr)
        print("Please re-run preprocessor with different dataset", file=sys.stderr)
        shutil.rmtree(OUTPUT)


if __name__ == '__main__':
    main()
