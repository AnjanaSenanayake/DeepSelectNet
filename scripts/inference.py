import glob
import sys
from timeit import default_timer as timer
from datetime import timedelta
from pathlib import Path

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from random import randrange
import pyslow5
import numpy as np
import click
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import *
from core.Logger import Logger

CUTOFF = 1500
SUB_SAMPLE_SIZE = 3000
SAMPLING_C0 = 1
BATCH = 1
IS_PICO = True
LABEL = 1
MAD_SCORE = 3
REPEATED = False
OUTPUT = 'predicts.txt'


@click.command()
@click.option('--saved_model', '-model', help='The model directory path', type=click.Path(exists=True))
@click.option('--slow5_dir', '-s5', help='path to slow5 directory', type=click.Path(exists=True))
@click.option('--batch', '-b', default=1, help='Batch size')
@click.option('--cutoff', '-c', default=1500, help='read signal cutoff value')
@click.option('--subsample_size', '-sz', default=3000, help='read signal sample size')
@click.option('--sampling_coefficient', '-sco', default=1, help='subsampling coefficient', type=int)
@click.option('--pico', '-pico', default=True, help='enable/disable pico conversion', type=bool)
@click.option('--label', '-lb', default=1, help='label of the train set', type=int)
@click.option('--mad', '-mad', default=3, help='mad value', type=float)
@click.option('--repeated_norm', '-rep', default=False, help='repeated normalization or not', type=bool)
@click.option('--output', '-o', help='output directory path', type=click.Path(exists=False))
def main(saved_model, slow5_dir, cutoff, subsample_size, sampling_coefficient, batch, pico, label, mad, repeated_norm,
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

    sys.stdout = Logger(OUTPUT)

    n_features = 1
    in_shape = (SUB_SAMPLE_SIZE, n_features)
    print(in_shape)

    '''evaluate the model'''
    model = keras.models.load_model(saved_model)
    model.compile(loss=BinaryCrossentropy().name, optimizer=tf.keras.optimizers.Adam(),
                  metrics=[BinaryAccuracy(), Precision(), Recall()])

    inference_model = model

    true_predicts = 0
    false_predicts = 0
    predicted_read_count = 0
    read_count = 0
    test_x_array = []
    accuracy = 0
    rejected_reads = 0
    per_read_time = 0
    per_read_prediction_time = 0

    start = timer()
    read_start = timer()
    s5 = pyslow5.Open(slow5_dir, 'r')
    _, num_reads = s5.get_read_ids()
    for read in s5.seq_reads(pA=IS_PICO):
        raw_read = read['signal']
        read_count += 1
        if (read['len_raw_signal'] - CUTOFF) > SUB_SAMPLE_SIZE:
            effective_read = raw_read[CUTOFF:]
            start_idx = randrange(len(effective_read) - SUB_SAMPLE_SIZE)
            end_idx = start_idx + SUB_SAMPLE_SIZE
            test_x = effective_read[start_idx:end_idx]
            # test_x = modified_zscore(test_x)
            # test_x = test_x.reshape(1, SUB_SAMPLE_SIZE, n_features)
            test_x = np.asarray(test_x, dtype=np.float32)
            test_x_array.append(test_x)

            if read_count % BATCH == 0:
                test_x = modified_zscore(test_x_array)
                read_prediction_start = timer()
                predicted_ys = inference_model.predict(test_x, batch_size=BATCH)
                read_prediction_end = timer()
                test_x_array = []

                for predicted_y in predicted_ys:
                    if predicted_y > 0.5:
                        rounded_predicted_y = 1
                    else:
                        rounded_predicted_y = 0

                    if rounded_predicted_y == LABEL:
                        true_predicts += 1
                    else:
                        false_predicts += 1
                    predicted_read_count += 1
                    per_read_time = (read_prediction_end - read_start) / BATCH
                    per_read_prediction_time = (read_prediction_end - read_prediction_start) / BATCH
                    accuracy = (true_predicts * 100) / (
                            true_predicts + false_predicts + tf.keras.backend.epsilon())
                    print("predicted: ", rounded_predicted_y, "Actual: ", LABEL, "Confidence: ", round(predicted_y[0] * 100, 4), "Accuracy: ", accuracy, "reads: ", predicted_read_count,
                          "/", num_reads)
        else:
            rejected_reads += 1
    end = timer()
    print("Total Time Elapsed: {}".format(str(timedelta(seconds=end - start))))
    print("Per Read Time Elapsed: {}".format(str(timedelta(seconds=per_read_time))))
    print("Per Read Prediction Time Elapsed: {}".format(str(timedelta(seconds=per_read_prediction_time))))
    print("True Predicts: ", true_predicts)
    print("False Predicts: ", false_predicts)
    print("Predicts Accuracy: ", accuracy, "%")
    print("Rejected Reads: ", rejected_reads)
    sys.stdout.close()


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data, axis=1)
    dev_from_med = np.array(data) - np.expand_dims(median, axis=1)
    mad = np.median(np.abs(dev_from_med), axis=1)
    mad_score = dev_from_med / (consistency_correction * np.expand_dims(mad, axis=1))

    x = np.where(np.abs(mad_score) > MAD_SCORE)
    # x = x[0]

    while True:
        if len(x) > 0:
            for i in range(x[0].shape[0]):
                if x[1][i] == 0:
                    # mad_score[x[i]] = mad_score[x[i] + 1]
                    mad_score[x[0][i], x[1][i]] = mad_score[x[0][i], x[1][i] + 1]
                elif x[1][i] == SUB_SAMPLE_SIZE - 1:
                    # mad_score[x[i]] = mad_score[x[i] - 1]
                    mad_score[x[0][i], x[1][i]] = mad_score[x[0][i], x[1][i] - 1]
                else:
                    # mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
                    mad_score[x[0][i], x[1][i]] = (mad_score[x[0][i], x[1][i] - 1] + mad_score[
                        x[0][i], x[1][i] + 1]) / 2
        else:
            break

        if REPEATED:
            x = np.where(np.abs(mad_score) > MAD_SCORE)
            x = x[0]
        else:
            break

    return mad_score


if __name__ == '__main__':
    main()
