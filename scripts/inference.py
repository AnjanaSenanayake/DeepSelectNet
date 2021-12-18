import glob
import sys
from random import randrange
import numpy as np
from os import walk
import click
import tensorflow as tf
from ont_fast5_api.fast5_interface import *
from tensorflow import keras
from tensorflow.keras.metrics import *
from ResNet import RESNET
from Logger import Logger

CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
SAMPLING_C0 = 1
BATCH = 1000
IS_PICO = True
LABEL = 1
MAD_SCORE = 3
REPEATED = False
OUTPUT = 'predicts.txt'


@click.command()
@click.option('--saved_model', '-model', help='The model directory path', type=click.Path(exists=True))
@click.option('--fast5_dir', '-f5', help='path to fast5 directory', type=click.Path(exists=True))
@click.option('--batch', '-b', default=1, help='Batch size')
@click.option('--cutoff', '-c', default=1500, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=3000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=1, help='subsampling coefficient', type=int)
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--label', '-lb', default=1, help='label of the train set', type=int)
@click.option('--mad', '-mad', default=3, help='mad value', type=int)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
@click.option('--output', '-o', help='npy output directory path', type=click.Path(exists=False))
def main(saved_model, fast5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, label, mad, repeated_norm,
         output):
    global CUTOFF, SUB_SAMPLE_SIZE, SAMPLING_C0, BATCH, IS_PICO, LABEL, OUTPUT, MAD_SCORE, REPEATED
    CUTOFF = cutoff
    SUB_SAMPLE_SIZE = subsample_size
    SAMPLING_C0 = sampling_coefficient
    BATCH = batch
    IS_PICO = pico
    LABEL = label
    MAD_SCORE = mad
    REPEATED = repeated_norm
    OUTPUT = output
    _, _, test_files = next(walk(fast5_dir))

    sys.stdout = Logger(OUTPUT)

    n_features = 1
    in_shape = (SUB_SAMPLE_SIZE, n_features)
    print(in_shape)

    '''evaluate the model'''
    model = keras.models.load_model(saved_model)
    model.compile(loss=BinaryCrossentropy().name, optimizer=tf.keras.optimizers.Adam(),
                  metrics=[BinaryAccuracy(), Precision(), Recall()])
    # inference_model = RESNET(input_shape=in_shape, nb_classes=1, is_train=False).model
    # inference_model.set_weights(model.get_weights())
    # inference_model.compile(loss=BinaryCrossentropy().name, optimizer=tf.keras.optimizers.Adam(),
    #                         metrics=[BinaryAccuracy(), Precision(), Recall()])
    inference_model = model

    # test_generator = DataGenerator(test_files, test_set, batch_size=batch, dim=in_shape, is_train=False)
    # scores = inference_model.evaluate(test_generator, verbose=1)
    #
    # for i in range(len(inference_model.metrics)):
    #     print("%s: %.2f%%" % (inference_model.metrics_names[i], scores[i]*100))

    true_predicts = 0
    false_predicts = 0
    f5_count = 0
    accuracy = 0
    rejected_reads = 0

    for fileNM in glob.glob(fast5_dir + '/*.' + ".fast5"):
        f5_count += 1
        file_path = fileNM
        with get_fast5_file(file_path, mode="r") as f5:
            for read in f5.get_reads():
                raw_data = read.get_raw_data(scale=IS_PICO)

                read_normalized = modified_zscore(raw_data)

                if (len(read_normalized) - CUTOFF) > SUB_SAMPLE_SIZE:
                    effective_read = read_normalized[CUTOFF:]

                    for i in range(SAMPLING_C0):
                        start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
                        end_idx = start_idx + SUB_SAMPLE_SIZE
                        test_x = effective_read[start_idx:end_idx]
                        test_x = test_x.reshape(1, SUB_SAMPLE_SIZE, n_features)
                        predicted_y = inference_model.predict(test_x, batch_size=BATCH)
                        if predicted_y > 0.5:
                            rounded_predicted_y = 1
                        else:
                            rounded_predicted_y = 0
                        if rounded_predicted_y == LABEL:
                            true_predicts += 1
                        else:
                            false_predicts += 1
                        accuracy = (true_predicts * 100) / (
                                    true_predicts + false_predicts + tf.keras.backend.epsilon())
                        print("predicted: ", rounded_predicted_y, "Actual: ", LABEL, "Confidence: ",
                              round(predicted_y[0][0] * 100, 4), "Accuracy: ", accuracy, "reads: ", f5_count,
                              "/",
                              len(test_files))
                else:
                    rejected_reads += 1
    print("True Predicts: ", true_predicts)
    print("False Predicts: ", false_predicts)
    print("Predicts Accuracy: ", accuracy, "%")
    print("Rejected Reads: ", rejected_reads)
    sys.stdout.close()


def convert_to_pico(raw_read, fh5):
    _range = int(fh5.channel_meta['range'])
    _offset = int(fh5.channel_meta['offset'])
    _digitisation = int(fh5.channel_meta['digitisation'])
    arr = np.zeros(raw_read.shape, dtype=np.float32)
    for index in range(len(raw_read)):
        arr[index] = (raw_read[index] + _offset) * (_range / _digitisation)
    return arr


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)
    dev_from_med = np.array(data) - median
    mad = np.median(np.abs(dev_from_med))
    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    x = x[0]

    while True:
        if len(x) > 0:
            # print(file, mad_score[x[0]])
            for i in range(len(x)):
                if x[i] == 0:
                    mad_score[x[i]] = mad_score[x[i] + 1]
                elif x[i] == len(mad_score) - 1:
                    mad_score[x[i]] = mad_score[x[i] - 1]
                else:
                    mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2

        x = np.where(np.abs(mad_score) > MAD_SCORE)
        x = x[0]
        if ~REPEATED or len(x) <= 0:
            break

    return mad_score


if __name__ == '__main__':
    main()
