import os
import sys
from os import walk
from datetime import datetime
import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback
from FCN import FCN
from ResNet import RESNET
from InceptionNet import InceptionNet
from TransformerNet import TransformerNet
from NPYDataGenerator import DataGenerator
from sklearn.model_selection import ShuffleSplit
import click
from Logger import Logger

cv_scores = []
CLASSIFIER = 'ResNet'
ACCURACY_METRIC = tf.keras.metrics.BinaryAccuracy().name
LOSS_FUNC = tf.keras.metrics.BinaryCrossentropy().name
IS_OCC = True
IS_ONE_HOT = False
TRAIN_SIZE = 0
VALIDATION_SIZE = 0
OUTPUT_PATH = 'trained_model_' + datetime.now().strftime("%m_%d_%Y")
train_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}
validation_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}


class ValidationCallback(Callback):
    def __init__(self, in_shape, model, val_generator):
        super().__init__()
        self.in_shape = in_shape
        self.model = model
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs={}):
        global LOSS_FUNC
        print("Validation Set Evaluation for Epoch ", epoch + 1)

        inference_model = get_classifier(self.in_shape, is_train=False)
        inference_model.set_weights(self.model.get_weights())
        inference_model.compile(loss=LOSS_FUNC,
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                         tf.keras.metrics.Recall()])
        val_scores = inference_model.evaluate(self.val_generator, verbose=1)

        for i in range(len(inference_model.metrics)):
            print("validation %s: %.2f%%" % (inference_model.metrics_names[i], val_scores[i] * 100))
            validation_history[list(validation_history.keys())[i]].append(val_scores[i])


def get_classifier(in_shape, nb_classes=1, is_train=True):
    model = None
    if CLASSIFIER == 'FCN':
        model = FCN(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif CLASSIFIER == 'ResNet':
        model = RESNET(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif CLASSIFIER == 'InceptionNet':
        model = InceptionNet(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif CLASSIFIER == 'TransformerNet':
        model = TransformerNet(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model

    model.compile(loss=LOSS_FUNC, optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])
    return model


def set_loss_func(loss_func):
    global ACCURACY_METRIC, LOSS_FUNC
    if loss_func == 'bc':
        ACCURACY_METRIC = tf.keras.metrics.BinaryAccuracy().name
        LOSS_FUNC = tf.keras.metrics.BinaryCrossentropy().name
    elif loss_func == 'cc':
        ACCURACY_METRIC = tf.keras.metrics.CategoricalAccuracy().name
        LOSS_FUNC = tf.keras.metrics.CategoricalCrossentropy().name
    elif loss_func == 'scc':
        ACCURACY_METRIC = tf.keras.metrics.SparseCategoricalAccuracy().name
        LOSS_FUNC = tf.keras.metrics.SparseCategoricalCrossentropy().name


def wrap_generator(generator):
    while True:
        x, y = next(generator)
        if IS_OCC:
            if IS_ONE_HOT:
                y = tf.keras.utils.to_categorical(y, num_classes=2)
                zeros = tf.zeros_like(y) + tf.constant([1., 0.])
            else:
                zeros = tf.zeros_like(y)
            y = tf.concat([y, zeros], axis=0)
        elif IS_ONE_HOT:
            y = tf.keras.utils.to_categorical(y, num_classes=2)
        yield x, y


def wrap_predict_generator(generator):
    for i in range(len(generator)):
        x, y = next(generator)
        yield x


def draw_plots(epochs, k):
    # plot accuracy history
    x = np.arange(1, epochs + 1, 1)
    plt.figure(0)
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['accuracy'], label='train_' + str(k))
    plt.plot(x, validation_history['accuracy'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/accuracy.png')

    # plot precision history
    plt.figure(2)
    plt.title('Train vs Validation Precision')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['precision'], label='train_' + str(k))
    plt.plot(x, validation_history['precision'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/precision.png')

    # plot recall history
    plt.figure(1)
    plt.title('Train vs Validation Recall')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['recall'], label='train_' + str(k))
    plt.plot(x, validation_history['recall'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/recall.png')

    # plot loss history
    plt.figure(3)
    plt.title('Train vs Validation Loss')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['loss'], label='train_' + str(k))
    plt.plot(x, validation_history['loss'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/loss.png')


@click.command()
@click.option('--dataset', '-d', default=None, help='The dataset directory path',
              type=click.Path(exists=True))
@click.option('--classifier', '-c', default='ResNet', help='classifier, default=ResNet', type=str)
@click.option('--loss_func', '-lf', default='bc', help='The loss function of the model[bc, cc, scc], default=bc',
              type=str)
@click.option('--split_ratio', '-s', default=0.75, help='The datasets split ratio, default=0.75', type=float)
@click.option('--is_occ', '-occ', default=False, help='Perform one class classification, default=True', type=bool)
@click.option('--is_one_hot', '-oh', default=False, help='One hot encode labels, default=False', type=bool)
@click.option('--folds', '-k', default=10, help='k fold value, default=10', type=int)
@click.option('--epochs', '-e', default=10, help='Number of epochs, default=10', type=int)
@click.option('--batch', '-b', default=1000, help='Batch size, default=1000', type=int)
@click.option('--output', '-o', help='The output directory path', type=click.Path(exists=False))
def main(dataset, classifier, loss_func, split_ratio, is_occ, is_one_hot, folds, epochs, batch, output):
    global CLASSIFIER, TRAIN_SIZE, VALIDATION_SIZE, IS_OCC, IS_ONE_HOT, OUTPUT_PATH
    CLASSIFIER = classifier
    IS_OCC = is_occ
    IS_ONE_HOT = is_one_hot
    if output:
        OUTPUT_PATH = output

    """load the dataset"""
    npy_files = None
    data = None
    pos_data = None
    neg_data = None
    n_steps = None
    dataset_size = None

    if dataset:
        _, _, npy_files = next(walk(dataset))
        pos_data = np.load(dataset + '/' + npy_files[0])
        neg_data = np.load(dataset + '/' + npy_files[1])
        n_steps = pos_data.shape[1] - 1
        dataset_size = len(pos_data) + len(neg_data)
    else:
        print("No dataset found")
        exit()

    n_features = 1
    in_shape = (n_steps, n_features)
    TRAIN_SIZE = dataset_size * split_ratio
    VALIDATION_SIZE = dataset_size - TRAIN_SIZE

    os.mkdir(OUTPUT_PATH)
    sys.stdout = Logger(OUTPUT_PATH + '/train-log.txt')

    '''training the model'''
    k = 1
    idx = np.arange(1, (dataset_size / 2) + 1, 1)
    k_fold = ShuffleSplit(n_splits=folds, train_size=split_ratio, random_state=0)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="binary_accuracy", factor=0.1, patience=5, mode="max",
                                                     min_lr=0, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=15, verbose=1,
                                                      restore_best_weights=True)
    # validation_callback = ValidationCallback(in_shape, model, validation)

    print("****Train Configs****")
    print("Classifier: ", classifier)
    print("Data Set Size: ", dataset_size)
    print("Batch Size: ", batch)
    print("input shape: ", in_shape)
    print("Epochs: ", epochs)
    print("Train split: {} Validation split: {}".format(TRAIN_SIZE, VALIDATION_SIZE))
    print("Cross Folds: ", folds)
    print("Loss function: ", loss_func)
    print("Split Ratio: ", split_ratio)
    print("Is OCC: ", is_occ)
    print("Is One Hot: ", is_one_hot)
    print("LR Monitor: ", reduce_lr.monitor)
    print("LR Factor: ", reduce_lr.factor)
    print("LR Patience: ", reduce_lr.patience)
    print("Early Stopping Monitor: ", early_stopping.monitor)
    print("Early Stopping Patience: ", early_stopping.patience)

    for train_idx, validation_idx in k_fold.split(idx):
        pos_train = pos_data[train_idx]
        pos_validation = pos_data[validation_idx]
        neg_train = neg_data[train_idx]
        neg_validation = neg_data[validation_idx]
        train = np.concatenate((pos_train, neg_train), axis=0)
        validation = np.concatenate((pos_validation, neg_validation), axis=0)

        # np.random.shuffle(train)
        # np.random.shuffle(validation)

        print("\nfold " + str(k) + " in training...")

        """create the model"""
        set_loss_func(loss_func)
        model = get_classifier(in_shape=in_shape, is_train=False)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_PATH + '/checkpoints_' + str(k),
                                                              monitor='val_binary_accuracy',
                                                              save_best_only=True, verbose=1)

        history = model.fit(
            x=train[:, :-1],
            y=train[:, -1],
            batch_size=batch,
            # validation_split=0.5,
            validation_data=(validation[:, :-1], validation[:, -1]),
            validation_batch_size=batch,
            epochs=epochs,
            callbacks=[reduce_lr, early_stopping, model_checkpoint],
            verbose=1)

        for i in range(len(model.metrics)):
            train_history[list(train_history.keys())[i]] = history.history[model.metrics_names[i]]
            validation_history[list(validation_history.keys())[i]] = history.history['val_' + model.metrics_names[i]]

        best_val_accuracy = round(early_stopping.best * 100)
        # best_val_accuracy = round(history.history['val_binary_accuracy'] * 100)
        cv_scores.append(best_val_accuracy)

        '''save model'''
        model_path = OUTPUT_PATH + "/model_" + str(best_val_accuracy) + '_k' + str(k)
        model.save(model_path)
        print("Best checkpoint: ", model_checkpoint.best)
        print("Saved model to disk at ", model_path)

        ep = epochs
        if early_stopping.stopped_epoch:
            ep = early_stopping.stopped_epoch + 1

        draw_plots(ep, k)
        train_history['loss'].clear()
        train_history['accuracy'].clear()
        train_history['precision'].clear()
        train_history['recall'].clear()
        validation_history['loss'] = []
        validation_history['accuracy'] = []
        validation_history['precision'] = []
        validation_history['recall'] = []
        k += 1

    # plot accuracy over k folds
    plt.figure(4)
    plt.title('Validation Accuracy')
    plt.xlabel('No of folds(k)')
    plt.ylabel('Percentage')
    plt.plot(numpy.arange(1, k, 1), cv_scores, label='validation')
    plt.legend()
    plt.savefig(OUTPUT_PATH + '/validation_accuracies_over_folds.png')

    for i, sc in enumerate(cv_scores):
        print("%d fold accuracy: %.2f%%" % (i + 1, sc))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    sys.stdout.close()


if __name__ == '__main__':
    main()
