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

'''evaluate the model'''
test_generator = Data_Generator(test_files, batch_size=64)
scores = model.evaluate(test_generator, verbose=0)
print("model evaluation: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
