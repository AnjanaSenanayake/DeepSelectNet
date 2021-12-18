import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib

matplotlib.use('agg')


class RESNET:

    def __init__(self, input_shape, nb_classes, verbose=False, build=True, is_train=True):
        if build:
            self.model = self.build_model(input_shape, nb_classes, is_train)
            if verbose:
                self.model.summary()
            self.verbose = verbose
        return

    def conv_pass_1(self, x, n_feature_maps=20):
        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='causal', strides=1)(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        return x1

    def conv_pass_2(self, x, n_feature_maps=20, strides=1):
        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='causal', strides=strides)(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        return x1

    def conv_pass_3(self, x, n_feature_maps=20):
        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='causal', strides=1)(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        return x1

    def make_layer(self, x, filters, blocks, strides=1):
        filter_1 = 20
        down_sample = None

        if strides != 1 or filter_1 != filters:
            down_sample = True

        x = self.conv_pass_1(x, filters)
        x = self.conv_pass_2(x, filters, strides=strides)
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='causal', strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if down_sample:
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=strides)(x)
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        if down_sample:
            filter_1 = filters

        for _ in range(1, blocks):
            x = self.conv_pass_1(x, filters)
            x = self.conv_pass_2(x, filters)
            x = self.conv_pass_3(x, filters)
        return x

    def build_model(self, input_shape, nb_classes, is_train):

        input_layer = tf.keras.layers.Input(input_shape)

        # BLOCK 1
        x = tf.keras.layers.Conv1D(filters=20, kernel_size=19, padding='causal', strides=3)(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2, padding='valid', strides=2)(x)

        # LAYERS
        x = tf.keras.layers.Dropout(0.10)(x)
        x = self.make_layer(x, filters=20, blocks=2)
        x = tf.keras.layers.Dropout(0.10)(x)
        x = self.make_layer(x, filters=30, blocks=2, strides=2)
        x = tf.keras.layers.Dropout(0.10)(x)
        x = self.make_layer(x, filters=45, blocks=2, strides=2)
        x = tf.keras.layers.Dropout(0.10)(x)
        x = self.make_layer(x, filters=67, blocks=2, strides=2)

        # FINAL
        x = tf.keras.layers.AveragePooling1D(1)(x)
        gap_layer = tf.keras.layers.Flatten()(x)

        noise_layer = K.zeros_like(gap_layer)
        noise_layer = tf.keras.layers.GaussianNoise(10)(noise_layer)

        if is_train:
            x = tf.keras.layers.Lambda(lambda z: tf.concat(z, axis=0))([gap_layer, noise_layer])
        else:
            x = gap_layer

        output_layer = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model
