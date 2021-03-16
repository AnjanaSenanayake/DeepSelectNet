import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

class Data_Generator(Sequence) :
  
  def __init__(self, X, y, batch_size) :
    self.X = X
    self.y = y
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.X) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.X[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array(batch_x), np.array(batch_y)