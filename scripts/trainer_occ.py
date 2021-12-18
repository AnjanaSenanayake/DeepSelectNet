from os import walk
import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback
from FCN import FCN
from ResNet import RESNET
from InceptionNet import InceptionNet
from NPYDataGenerator import DataGenerator
from sklearn.model_selection import ShuffleSplit
import click

cv_scores = []
CLASSIFIER = 'ResNet'
train_history = {'loss': [], 'binary_accuracy': [], 'precision': [], 'recall': []}
validation_history = {'loss': [], 'binary_accuracy': [], 'precision': [], 'recall': []}


class ValidationCallback(Callback):
    def __init__(self, in_shape, model, pos_generator, neg_generator=None):
        super().__init__()
        self.in_shape = in_shape
        self.model = model
        self.pos_generator = pos_generator
        self.neg_generator = neg_generator

    def on_epoch_end(self, epoch, logs={}):
        print("Validation Set Evaluation for Epoch ", epoch + 1)

        inference_model = get_classifier(self.in_shape, is_train=False)
        inference_model.set_weights(self.model.get_weights())
        inference_model.compile(loss='binary_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                                         tf.keras.metrics.Recall()])

        val_scores = inference_model.evaluate(wrap_evaluate_generator(self.pos_generator, self.neg_generator),
                                              verbose=1)
        for i in range(len(inference_model.metrics)):
            print("validation %s: %.2f%%" % (inference_model.metrics_names[i], val_scores[i] * 100))
            validation_history[list(validation_history.keys())[i]].append(val_scores[i])

        print("/n/n")


def get_classifier(in_shape, nb_classes=1, is_train=False):
    model = None
    if CLASSIFIER == 'FCN':
        model = FCN(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif CLASSIFIER == 'ResNet':
        model = RESNET(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif CLASSIFIER == 'InceptionNet':
        model = InceptionNet(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    return model


def wrap_train_generator(pos_generator, neg_generator=None, is_one_class=False):
    x = None
    y = None
    pos_len = len(pos_generator)
    if is_one_class:
        for i in range(pos_len):
            x, y = next(pos_generator)
            # y = tf.keras.utils.to_categorical(y, num_classes=2)
            zeros = tf.zeros_like(y)
            y = tf.concat([y, zeros], axis=0)
            print('HERE1')
            yield x, y
    elif neg_generator:
        neg_len = len(neg_generator)
        print('HERE2', neg_len)
        if pos_len > neg_len:
            for i in range(pos_len):
                x_pos, y_pos = next(pos_generator)
                # y_pos = tf.keras.utils.to_categorical(y_pos, num_classes=2)
                if i < neg_len:
                    x_neg, y_neg = next(neg_generator)
                    # y_neg = tf.keras.utils.to_categorical(y_neg, num_classes=2)
                    x = tf.concat([x_pos, x_neg], axis=0)
                    y = tf.concat([y_pos, y_neg], axis=0)
                yield x, y
        else:
            print('HERE4')
            for i in range(neg_len):
                x, y = next(neg_generator)
                # y_neg = tf.keras.utils.to_categorical(y_neg, num_classes=2)
                # if i < pos_len:
                #     x_pos, y_pos = next(pos_generator)
                #     # y_pos = tf.keras.utils.to_categorical(y_pos, num_classes=2)
                #     x = tf.concat([x_pos, x], axis=0)
                #     y = tf.concat([y_pos, y], axis=0)
                yield x, y


def wrap_evaluate_generator(pos_generator, neg_generator):
    x = None
    y = None
    pos_len = len(pos_generator)
    if neg_generator:
        neg_len = len(neg_generator)
        if pos_len > neg_len:
            for i in range(pos_len):
                x_pos, y_pos = next(pos_generator)
                # y_pos = tf.keras.utils.to_categorical(y_pos, num_classes=2)
                for j in range(neg_len):
                    x_neg, y_neg = next(neg_generator)
                    # y_neg = tf.keras.utils.to_categorical(y_neg, num_classes=2)
                    x = tf.concat([x_pos, x_neg], axis=0)
                    y = tf.concat([y_pos, y_neg], axis=0)
        else:
            for i in range(neg_len):
                x_neg, y_neg = next(neg_generator)
                # y_neg = tf.keras.utils.to_categorical(y_neg, num_classes=2)
                for j in range(pos_len):
                    x_pos, y_pos = next(pos_generator)
                    # y_pos = tf.keras.utils.to_categorical(y_pos, num_classes=2)
                    x = tf.concat([x_pos, x_neg], axis=0)
                    y = tf.concat([y_pos, y_neg], axis=0)
    else:
        for i in range(pos_len):
            x, y = next(pos_generator)
            # y = tf.keras.utils.to_categorical(y, num_classes=2)
    yield x, y


def wrap_predict_generator(generator_len, generator):
    for i in range(generator_len):
        x, y = next(generator)
        yield x


def train_model(model, in_shape, batch, epochs, k, positive_dataset, positives, pos_files, negative_dataset=None,
                negatives=None, neg_files=None):
    print("\nfold " + str(k) + " in training...")

    pos_train, pos_validation = positives
    neg_train, neg_validation = negatives
    train_pos = [pos_files[idx] for idx in pos_train]
    validation_pos = [pos_files[idx] for idx in pos_validation]
    if neg_files:
        train_neg = [neg_files[idx] for idx in neg_train]
        validation_neg = [neg_files[idx] for idx in neg_validation]
    else:
        train_neg = []
        validation_neg = []

    train = train_pos + train_neg
    validation = validation_pos + validation_neg

    print("Train split: {} (positive:{} negative:{})".format(len(train), len(train_pos), len(train_neg)))
    print("Validation split: {} (positive:{} negative:{})".format(len(validation), len(validation_pos),
                                                                  len(validation_neg)))

    pos_train_generator = DataGenerator(train_pos, positive_dataset, batch_size=batch, dim=in_shape)
    pos_validation_generator = DataGenerator(validation_pos, positive_dataset, batch_size=batch, dim=in_shape)

    if neg_files:
        neg_train_generator = DataGenerator(train_neg, negative_dataset, batch_size=batch, dim=in_shape)
        neg_validation_generator = DataGenerator(validation_neg, negative_dataset, batch_size=batch,
                                                 dim=in_shape)
    else:
        neg_train_generator = None
        neg_validation_generator = None

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='binary_accuracy', factor=0.5, patience=50,
                                                     min_lr=0.0001)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='trained_model/checkpoints', monitor='loss',
                                                          save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
    validation_callback = ValidationCallback(in_shape, model, pos_validation_generator,
                                             neg_validation_generator)

    history = model.fit_generator(
        generator=wrap_train_generator(pos_train_generator, neg_train_generator, is_one_class=False),
        steps_per_epoch=len(train) // 1,
        epochs=epochs,
        callbacks=[reduce_lr, model_checkpoint, early_stopping, validation_callback],
        shuffle=True,
        verbose=1)

    for i in range(len(model.metrics)):
        train_history[list(train_history.keys())[i]] = history.history[model.metrics_names[i]]

    print(train_history)
    print(validation_history)
    cv_scores.append(validation_history['binary_accuracy'][-1] * 100)

    '''save model'''
    validation_accuracy = round(cv_scores[-1])
    model_path = "trained_model/model_" + str(validation_accuracy) + '_k' + str(k)
    model.save(model_path)
    print("Saved model to disk")

    draw_plots(epochs, 'trained_model', k)
    train_history['loss'] = []
    train_history['binary_accuracy'] = []
    train_history['precision'] = []
    train_history['recall'] = []
    validation_history['loss'] = []
    validation_history['binary_accuracy'] = []
    validation_history['precision'] = []
    validation_history['recall'] = []


def draw_plots(epochs, model_path, k):
    # plot accuracy history
    x = np.arange(1, epochs + 1, 1)
    plt.figure(0)
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['binary_accuracy'], label='train_' + str(k))
    plt.plot(x, validation_history['binary_accuracy'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(model_path + '/accuracy.png')

    # plot precision history
    plt.figure(2)
    plt.title('Train vs Validation Precision')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['precision'], label='train_' + str(k))
    plt.plot(x, validation_history['precision'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(model_path + '/precision.png')

    # plot recall history
    plt.figure(1)
    plt.title('Train vs Validation Recall')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['recall'], label='train_' + str(k))
    plt.plot(x, validation_history['recall'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(model_path + '/recall.png')

    # plot loss history
    plt.figure(3)
    plt.title('Train vs Validation Loss')
    plt.xlabel('No of Epochs')
    plt.ylabel('Percentage')
    plt.plot(x, train_history['loss'], label='train_' + str(k))
    plt.plot(x, validation_history['loss'], label='validation_' + str(k))
    plt.legend()
    plt.savefig(model_path + '/loss.png')


@click.command()
@click.option('--positive_dataset', '-pos', default=None, help='The positive dataset directory path',
              type=click.Path(exists=True))
@click.option('--negative_dataset', '-neg', default=None, help='The negative dataset directory path',
              type=click.Path(exists=True))
@click.option('--classifier', '-c', default='ResNet', help='classifier', type=str)
@click.option('--split_ratio', '-s', default=0.75, help='The datasets split ratio', type=float)
@click.option('--folds', '-k', default=10, help='k fold value', type=int)
@click.option('--epochs', '-e', default=10, help='Number of epochs')
def main(positive_dataset, negative_dataset, classifier, split_ratio, folds, epochs):
    global CLASSIFIER
    CLASSIFIER = classifier
    """load the dataset"""
    pos_files = None
    neg_files = None
    neg_folds = None
    pos_data = None
    neg_data = None
    batch = None
    n_steps = None
    dataset_size = None

    if positive_dataset:
        _, _, pos_files = next(walk(positive_dataset))
        np.random.shuffle(pos_files)
        pos_data = np.load(positive_dataset + '/' + pos_files[1])
        batch = pos_data.shape[0]
        n_steps = pos_data.shape[1] - 1
        dataset_size = len(pos_files)
    else:
        print("No positive dataset found")
        exit()

    if negative_dataset:
        _, _, neg_files = next(walk(negative_dataset))
        np.random.shuffle(neg_files)
        neg_data = np.load(positive_dataset + '/' + pos_files[1])
        dataset_size = dataset_size + len(neg_files)

        if batch != neg_data.shape[0] or n_steps != (neg_data.shape[1] - 1):
            print("Unmatched dimensions for positive & negative datasets\n")
            print("positive data: ", pos_data.shape)
            print("negative data: ", neg_data.shape)
            exit()

    n_features = 1
    in_shape = (n_steps, n_features)

    print("Data Set Size: ", dataset_size)
    print("batch size: ", batch)
    print("input shape: ", in_shape)

    """create the model"""
    model = get_classifier(in_shape=in_shape)

    '''training the model'''
    k_fold = ShuffleSplit(n_splits=folds, train_size=split_ratio, random_state=0)
    k = 1

    if negative_dataset:
        for positives, negatives in zip(k_fold.split(pos_files), k_fold.split(neg_files)):
            train_model(model, in_shape, batch, epochs, k, positive_dataset, positives, pos_files, negative_dataset,
                        negatives, neg_files)
            k += 1
    else:
        for positives in k_fold.split(pos_files):
            train_model(model, in_shape, batch, epochs, k, positive_dataset, positives, pos_files,
                        negative_dataset=None,
                        negatives=None, neg_files=None)
            k += 1

    # plot accuracy over k folds
    plt.figure(4)
    plt.title('Validation Accuracy')
    plt.xlabel('No of folds(k)')
    plt.ylabel('Percentage')
    plt.plot(numpy.arange(1, k, 1), cv_scores, label='validation')
    plt.legend()
    plt.savefig('trained_model/validation_accuracy_over_folds.png')

    for i, sc in enumerate(cv_scores):
        print("%d fold accuracy: %.2f%%" % (i + 1, sc))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))


if __name__ == '__main__':
    main()
