import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import TimeseriesGenerator

#import cv2
import random
import itertools
import os

SHAPE = (4000)
batch_size = 32

def get_model(train=True):
    
    pre_process = Lambda(preprocess_input)
    
    inp = Input(SHAPE)
    inp_out = pre_process(GaussianNoise(0.1)(inp))
    
    noise = Lambda(tf.zeros_like)(inp_out)
    noise = GaussianNoise(0.1)(noise)

    if train:
        x = Lambda(lambda z: tf.concat(z, axis=0))([inp_out,noise])
        x = Activation('relu')(x)
    else:
        x = inp_out
        
    x = Dense(input_dim=32, units=4000, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(Adam(lr=1e-4), loss='binary_crossentropy')
    
    return model


data = np.array([[i] for i in range(40000)])
targets = np.array([[i] for i in range(40000)])
data_gen = TimeseriesGenerator(data, targets, length=4000, sampling_rate=1, batch_size=32)


tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

es = EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=5)

model = get_model()
model.fit(data_gen, steps_per_epoch=data_gen.samples/data_gen.batch_size, epochs=20)

