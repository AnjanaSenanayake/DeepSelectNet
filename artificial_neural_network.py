# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
tf.__version__


# Importing the dataset
dataset = pd.read_csv('/storage/e14317/covid/train_data.csv')
validation_dataset = pd.read_csv('/storage/e14317/zymo/Zymo-GridION-EVEN-BB-SN/train_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_val = validation_dataset.iloc[:,:-1].values
y_val = validation_dataset.iloc[:,-1].values

mu = np.mean(X[:,:])
stdev = np.std(X[:,:])
noise = np.random.normal(mu, stdev, X.shape)
generated_X = (X+noise)/2
generated_y = np.zeros(y.shape,float)

X = np.concatenate((X, generated_X), axis=0)
y = np.concatenate((y, generated_y), axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)


# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=500, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=200, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 128, epochs = 100)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_val_pred = ann.predict(X_val)
y_pred = (y_pred > 0.5)
y_val_pred = (y_val_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm = confusion_matrix(y_val, y_val_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_val, y_val_pred))

