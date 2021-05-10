import sys
import numpy as np
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from LocalConfigs import *
from data_generator import Data_Generator

test_files = []
with open(sys.argv[1]) as f:
    content = f.readlines()
for line in content:
    test_files.append(line.rstrip('\n'))

'''create the model'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(500,1)))
model.add(Dropout(0.2))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.LSTM(10))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''evaluate the model'''
test_generator = Data_Generator(test_files, batch_size=64)
scores = model.evaluate(test_generator, verbose=0)
print("model evaluation: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
