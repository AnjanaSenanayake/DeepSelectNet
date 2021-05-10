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

'''create the model'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=16, activation='relu', input_shape=(500,1)))
model.add(Dropout(0.2))
model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
model.add(tf.keras.layers.LSTM(10))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


'''training the model'''
train_generator = Data_Generator(train_partition, batch_size=128)
test_generator = Data_Generator(test_partition, batch_size=64)
model.fit_generator(generator=train_generator,
                  validation_data=test_generator,
                  steps_per_epoch=10,
                  epochs=100)


'''evaluate the model'''
scores = model.evaluate(test_generator, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


'''serialize model to JSON'''
model_json = model.to_json()
with open("model_" + str(scores[1]*100) + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
