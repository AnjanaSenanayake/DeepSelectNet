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

train_files = []
train_partition = []
test_partition = []
with open(NPY_LIST_FILE) as f:
    content = f.readlines()
for line in content:
    train_files.append(line.rstrip('\n'))

train_split = 0.7
test_split = 0.3
train_partition = train_files[0:int(len(train_files) * train_split)]
test_partition = train_files[int(len(train_files) * train_split):]

# create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(500,1)))
model.add(Dropout(0.2))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.LSTM(10))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

train_generator = Data_Generator(train_partition, batch_size=128)
test_generator = Data_Generator(test_partition, batch_size=64)
model.fit_generator(generator=train_generator,
                  validation_data=test_generator,
                  steps_per_epoch=6,
                  epochs=10)

# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
