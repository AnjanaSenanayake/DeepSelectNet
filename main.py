import gzip
import os
import shutil
import tarfile
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np


def load_model_values(file_name):
    model_dict = {}
    with open(file_name) as f:
        content = f.readlines()
    for line in content:
        line_tokens = line.split(',')
        model_dict.update({line_tokens[0]: line_tokens[1]})
    return model_dict


def read_fast5_files(fast5_dir):
    fast5_files = [file for file in listdir(fast5_dir) if isfile(join(fast5_dir, file))]
    print(fast5_files)
    for fast5_file in fast5_files:
        with gzip.open(fast5_dir + '/' + fast5_file, 'rb') as file_zip:
            with open(fast5_dir + '/' + fast5_file.split('.')[0] + '.txt', 'wb') as file_txt:
                shutil.copyfileobj(file_zip, file_txt)


def fast5_reads(file_name):
    read_string = ''
    print(file_name)
    with open(file_name) as f:
        content = f.readlines()
        print(content)
    for index in range(len(content)):
        if content[index].strip() == '+':
            read_string = read_string + content[index - 1].strip()
    return read_string


def ref_reads(file_name):
    reference_string = ''
    print(file_name)
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            reference_string = reference_string + line.strip()
        #print(content)
    return reference_string


def write_read_data_to_file(output_folder, read, signal_arr):
    with open(output_folder + "/read.txt", "a") as read_file:
        read_file.write(read)
        read_file.close()
    np.savetxt(output_folder + '/signal.csv', signal_arr, delimiter=',', fmt='%f')
    plt.plot(signal_arr, 'o')
    plt.ylabel('Current Mean Level')
    plt.xlabel('kmer')
    plt.savefig(output_folder + '/signal_image' + '.png')
    plt.show()


def call_raw_values(read, model):
    kmer = ''
    raw_values = np.zeros(len(read) - 5)
    for char_index in range(len(read)):
        if char_index + 6 <= len(read):
            kmer = read[char_index:char_index + 6]
            raw_values[char_index] = model.get(kmer)
            #print(kmer + ': ' + model.get(kmer))
    return raw_values


if __name__ == '__main__':
    nanopore_model = load_model_values('nanopore_model.txt')

    REF_MODE = True
    reads_string = ''

    if REF_MODE:
        dataset_dir = '../datasets/human_ref/'

        if os.path.exists('data'):
            shutil.rmtree('data')
        os.mkdir('data')

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(".fa"):
                    file_name = file.split('.')[0]
                    os.mkdir('data/' + file_name)
                    output_folder_path = 'data/' + file_name
                    file_path = dataset_dir + file
                    ref_string = ref_reads(file_path)
                    signal_values = call_raw_values(ref_string, nanopore_model)
                    write_read_data_to_file(output_folder_path, ref_string, signal_values)
    else:
        dataset_dir = '../datasets/'

        seq_files = [file for file in listdir(dataset_dir) if isfile(join(dataset_dir, file))]
        if os.path.exists(dataset_dir + '/temp'):
            shutil.rmtree(dataset_dir + '/temp')
        os.mkdir(dataset_dir + '/temp')

        if os.path.exists('data'):
            shutil.rmtree('data')
        os.mkdir('data')

        for seq_file in seq_files:
            seq_file_name = seq_file.split('.')[0]
            os.mkdir('data/' + seq_file_name)
            seq_zip_path = dataset_dir + '/' + seq_file
            seq_path = dataset_dir + '/temp/' + seq_file_name
            tar = tarfile.open(seq_zip_path, "r:")
            tar.extractall(seq_path)
            tar.close()

            for root, dirs, files in os.walk(seq_path):
                for file in files:
                    if file.endswith((".fastq.gz", "fasta.gz")):
                        file_name = file.split('.')[0]
                        os.mkdir('data/' + seq_file_name + '/' + file_name)
                        output_folder_path = 'data/' + seq_file_name + '/' + file_name
                        fast5_zip_path = root + '/' + file
                        fast5_txt_path = root + '/' + file_name + '.txt'
                        with gzip.open(fast5_zip_path, 'rb') as file_zip:
                            with open(fast5_txt_path, 'wb') as file_txt:
                                shutil.copyfileobj(file_zip, file_txt)
                                reads_string = fast5_reads(fast5_txt_path)
                                signal_values = call_raw_values(reads_string, nanopore_model)
                                write_read_data_to_file(output_folder_path, reads_string, signal_values)
    # print(len(raw_values))
    # print(raw_values)
    # generate_signal(raw_values)
