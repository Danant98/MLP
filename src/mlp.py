#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from layer import dense

# Defining the MLP class
class MLP:

    def __init__(self, X:np.ndarray, Y:np.ndarray, layer_size:list, epochs:int = 100,
                 lr:float = 0.1, momentum:float = 0.5):
        # Adding ones in the first column to account for bias
        self.X =  self.__normalize(X)
        self.Y = Y.squeeze()
        # One Hot encoding input y labels
        self.__OneHot()
        self.labels = np.unique(self.Y)
        # Setting learning rate, momentum and max number of epochs
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        # Size of each layer, that is the number of nodes in each layer
        self.layer_size = [X.shape[1]] + layer_size + [self.labels.shape[0]]
        # Initializing the network
        self.__create_network()
        # Error array containing cost function
        self.error = np.zeros(epochs)
    
    def __normalize(self, X:np.ndarray):
        """
        Normalize the features to N(0, 1)
        """
        return (X - np.mean(X, axis = 1)) / np.std(X, axis = 1)

    @staticmethod
    def __OneHot(self):
        """
        One Hot encoding the labels for cost function
        """
        self.Y_onehot = np.zeros((self.Y.shape[0], self.labels.shape[0]))
        for i in range(self.Y.shape[0]):
            self.Y_onehot[i, self.Y[i] - 1] = 1
    
    @staticmethod
    def __create_network(self):
        """
        Creating the network based on inputed size with fixed input and output layer sizes
        """
        self.layers = []
        for i in range(len(self.layer_size - 1)):
            self.layers.append(dense(self.layer_size[i], self.layer_size[i + 1]))

    def train(self):
        """
        Training the network
        """
        for e in range(self.epochs):
            pass

    def predict(self):
        pass

