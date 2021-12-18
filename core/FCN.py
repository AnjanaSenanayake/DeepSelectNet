import tensorflow as tf
from tensorflow.keras import backend as K


class FCN:

    def __init__(self, input_shape, nb_classes, verbose=False, build=True, is_train=True):
        if build:
            self.model = self.build_model(input_shape, nb_classes, is_train)
            if verbose:
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes, is_train):
        input_layer = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        noise_layer = K.zeros_like(gap_layer)
        noise_layer = tf.keras.layers.GaussianNoise(0.3)(noise_layer)

        if is_train:
            x = tf.keras.layers.Lambda(lambda z: tf.concat(z, axis=0))([gap_layer, noise_layer])
        else:
            x = gap_layer

        output_layer = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model
