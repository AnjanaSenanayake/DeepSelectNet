import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from LocalConfigs import *

class Data_Generator(Sequence):
  
  '''Initialization'''
  def __init__(self, list_IDs, batch_size, dim=(4000,501), n_classes=2, shuffle=True):
        
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    

  '''Updates indexes after each epoch'''
  def on_epoch_end(self):
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    
  '''Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)'''
  def __data_generation(self, list_IDs_temp):
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          # print(ID)
          data = np.load(TRAIN_DIR + ID + '.npy')
          X = data[:, :-1]
          X = np.expand_dims(X, axis=2)
    
          # Store class
          y = data[:, -1]
    
      return X, y


  '''Denotes the number of batches per epoch'''
  def __len__(self):
      return int(np.floor(len(self.list_IDs) / self.batch_size))
  

  '''Generate one batch of data'''
  def __getitem__(self, index):
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return X, y
