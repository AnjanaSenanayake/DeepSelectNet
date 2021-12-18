from tensorflow.keras import *


class TransformerNet:

    def __init__(self, input_shape, nb_classes, verbose=False, build=True, is_train=True):
        if build:
            self.model = self.build_model(input_shape, nb_classes, mlp_units=[2], is_train=is_train)
            if verbose:
                self.model.summary()
            self.verbose = verbose
        return

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self, input_shape, nb_classes, head_size=1, num_heads=1, ff_dim=1,
                    num_transformer_blocks=1, mlp_units=None, dropout=0.25, mlp_dropout=0.4, is_train=False):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(nb_classes, activation="sigmoid")(x)
        return Model(inputs, outputs)
