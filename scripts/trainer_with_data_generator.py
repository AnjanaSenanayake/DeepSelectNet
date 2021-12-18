from os import walk
from  datetime import datetime
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
@click.option('--classifier', '-c', default='ResNet', help='classifier', type=str)
@click.option('--loss_func', '-lf', default='bc', help='The loss function of the model', type=str)
@click.option('--split_ratio', '-s', default=0.75, help='The datasets split ratio', type=float)
@click.option('--is_occ', '-occ', default=True, help='Perform one class classification', type=bool)
@click.option('--is_one_hot', '-oh', default=False, help='One hot encode labels', type=bool)
@click.option('--folds', '-k', default=10, help='k fold value', type=int)
@click.option('--epochs', '-e', default=10, help='Number of epochs')
def main(dataset, classifier, loss_func, split_ratio, is_occ, is_one_hot, folds, epochs):
    global CLASSIFIER, TRAIN_SIZE, VALIDATION_SIZE, IS_OCC, IS_ONE_HOT
    CLASSIFIER = classifier
    IS_OCC = is_occ
    IS_ONE_HOT = is_one_hot

    """load the dataset"""
    npy_files = None
    batch = None
    n_steps = None
    dataset_size = None

    if dataset:
        _, _, npy_files = next(walk(dataset))
        np.random.shuffle(npy_files)
        data = np.load(dataset + '/' + npy_files[1])
        batch = data.shape[0]
        n_steps = data.shape[1] - 1
        dataset_size = len(npy_files)
    else:
        print("No dataset found")
        exit()

    n_features = 1
    in_shape = (n_steps, n_features)

    print("Data Set Size: ", dataset_size)
    print("batch size: ", batch)
    print("input shape: ", in_shape)

    """create the model"""
    set_loss_func(loss_func)
    model = get_classifier(in_shape=in_shape, is_train=False)

    '''training the model'''
    k = 1
    k_fold = ShuffleSplit(n_splits=folds, train_size=split_ratio, random_state=0)
    for train_ids, validation_ids in k_fold.split(npy_files):
        print("\nfold " + str(k) + " in training...")
        TRAIN_SIZE = len(train_ids)
        VALIDATION_SIZE = len(validation_ids)
        train = [npy_files[idx] for idx in train_ids]
        validation = [npy_files[idx] for idx in validation_ids]

        print("Train split: {} Validation split: {}".format(TRAIN_SIZE, VALIDATION_SIZE))

        train_generator = DataGenerator(train, dataset, in_shape, batch)
        validation_generator = DataGenerator(validation, dataset, in_shape, batch)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=ACCURACY_METRIC, factor=0.5, patience=50,
                                                         min_lr=0.0001)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_PATH + '/checkpoints', monitor='loss',
                                                              save_best_only=True)
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1)
        validation_callback = ValidationCallback(in_shape, model, validation_generator)

        history = model.fit_generator(
            generator=wrap_generator(train_generator),
            steps_per_epoch=TRAIN_SIZE,
            validation_steps=VALIDATION_SIZE,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[reduce_lr, model_checkpoint],
            shuffle=True,
            verbose=1)

        for i in range(len(model.metrics)):
            train_history[list(train_history.keys())[i]] = history.history[model.metrics_names[i]]
            validation_history[list(validation_history.keys())[i]] = history.history['val_' + model.metrics_names[i]]

        cv_scores.append(validation_history['accuracy'][-1] * 100)

        '''save model'''
        validation_accuracy = round(cv_scores[-1])
        model_path = OUTPUT_PATH + "/model_" + str(validation_accuracy) + '_k' + str(k)
        model.save(model_path)
        print("Saved model to disk at ", model_path)

        draw_plots(epochs, k)
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


if __name__ == '__main__':
    main()
