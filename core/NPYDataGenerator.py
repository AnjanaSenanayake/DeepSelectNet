import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """Initialization"""

    def __init__(self, list_ids, dataset, dim, batch_size, n_classes=1, iterator=1, shuffle=True):
        self.list_ids = list_ids
        self.dataset = dataset
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.iterator = iterator
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_ids))
        self.on_epoch_end()
        self.n = 0

    '''Updates indexes after each epoch'''

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    '''Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)'''

    def __data_generation(self, list_ids_temp):
        # Initialization
        x = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y = np.zeros((self.batch_size, self.dim[1]), dtype=int)

        reads = np.load(self.dataset + '/' + list_ids_temp[0])
        for i in range(len(reads)):
            # store feature
            x[i, ] = reads[i][:-1].reshape(self.dim[0], self.dim[1])
            # Store class
            y[i, ] = reads[i][-1]
        # zeros = tf.zeros_like(y_bar)
        # y[i, ] = tf.concat([y_bar, zeros], axis=0)
        return x, y

    '''Denotes the number of batches per epoch'''

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.iterator))

    '''next iterator'''
    def __next__(self):
        if self.n >= self.__len__():
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    '''Generate one batch of data'''

    def __getitem__(self, index):
        # Generate indexes of the batch
        start_idx = index * self.iterator
        end_idx = (index + 1) * self.iterator

        if end_idx >= len(self.indexes):
            end_idx = len(self.indexes) - 1

        if start_idx == end_idx:
            indexes = [self.indexes[start_idx]]
        else:
            indexes = self.indexes[start_idx:end_idx]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        # print("\nBatch " + str(index) + " is loaded. Batch size " + str(len(list_ids_temp)))
        return x, y
