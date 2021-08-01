import os
from os import walk
import tensorflow as tf
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from datagenerator import DataGenerator
import click


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


@click.command()
@click.option('--dataset', '-d', help='The dataset directory path', type=click.Path(exists=True))
@click.option('--split', '-s', default=0.8, help='Train split of the dataset', type=float)
# @click.option('--batch', '-b', default=1, help='Batch size')
@click.option('--epochs', '-e', default=10, help='Number of epochs')
def main(dataset, split, epochs):
    """load the dataset"""
    _, _, train_files = next(walk(dataset))

    train_partition = train_files[0:int(len(train_files) * split)]
    validation_partition = train_files[int(len(train_files) * split):]

    data = np.load(dataset + train_partition[1])

    batch = data.shape[0]
    n_steps = data.shape[1] - 1
    n_features = 1
    in_shape = (n_steps, n_features)

    print("train partition: ", len(train_partition))
    print("validation partition: ", len(validation_partition))
    print("batch size: ", batch)
    print("input shape: ", in_shape)

    """create the model"""
    model = tf.keras.models.Sequential()
    model.add(cnn_layer)
    model.add(noise_layer)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.AveragePooling1D())
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.LSTM(10))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall_m, precision_m, f1_m])

    '''training the model'''
    train_generator = DataGenerator(train_partition, dataset, batch_size=batch, dim=in_shape)
    validation_generator = DataGenerator(validation_partition, dataset, batch_size=batch, dim=in_shape)
    history = model.fit_generator(generator=train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=len(train_partition) // 1,
                                  validation_steps=len(validation_partition) // 1,
                                  epochs=epochs,
                                  shuffle=True,
                                  verbose=1)

    '''evaluate the model'''
    print("Train Set Evaluation")
    scores = model.evaluate(train_generator, verbose=0)
    for i in range(len(model.metrics)):
        print("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

    print("Validation Set Evaluation")
    val_scores = model.evaluate(validation_generator, verbose=0)
    for i in range(len(model.metrics)):
        print("%s: %.2f%%" % (model.metrics_names[i], val_scores[i] * 100))

    '''save model'''
    training_accuracy = round(scores[1] * 100)
    model_path = "model_" + str(training_accuracy)
    model.save(model_path)
    print("Saved model to disk")

    # plot accuracy history
    plt.figure(0)
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.savefig(model_path + '/train_vs_validation_accuracy.png')

    # plot recall history
    plt.figure(1)
    plt.title('Train vs Validation Recall')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(history.history['recall_m'], label='train')
    plt.plot(history.history['val_recall_m'], label='test')
    plt.legend()
    plt.savefig(model_path + '/train_vs_validation_recall.png')

    # plot precision history
    plt.figure(2)
    plt.title('Train vs Validation Precision')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(history.history['precision_m'], label='train')
    plt.plot(history.history['val_precision_m'], label='test')
    plt.legend()
    plt.savefig(model_path + '/train_vs_validation_precision.png')

    # plot loss history
    plt.figure(3)
    plt.title('Train vs Validation Loss')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(model_path + '/train_vs_validation_loss.png')


if __name__ == '__main__':
    main()
