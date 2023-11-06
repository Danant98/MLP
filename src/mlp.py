
__author__ = 'Daniel Elisabethsønn Antonsen, UiT'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt

# Defining the MLP class
class MLP:

    def __init__(self, X:np.ndarray, Y:np.ndarray, 
                 learning_rate:float, momentum:float):
        self.X = X
        self.Y = Y.squeeze()
        self.labels = np.unique(self.Y)

    def OneHot(self):
        self.Y_onehot = np.zeros((self.Y.shape[0], self.labels.shape[0]))
        for i in range(self.Y.shape[0]):
            self.Y_onehot[i, self.Y[i] - 1] = 1

