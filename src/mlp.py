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
        self.X = X
        self.Y = Y.squeeze()
        # One Hot encoding input y labels
        self.OneHot()
        self.labels = np.unique(self.Y)
        # Setting learning rate, momentum and max number of epochs
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        # Size of each layer, that is the number of nodes in each layer
        self.layer_size = [X.shape[1]] + layer_size + [self.labels.shape[0]]
        # Set network
        self.create_network()
        # Error array containing loss
        self.error = np.zeros(epochs)

    def OneHot(self):
        self.Y_onehot = np.zeros((self.Y.shape[0], self.labels.shape[0]))
        for i in range(self.Y.shape[0]):
            self.Y_onehot[i, self.Y[i] - 1] = 1
    
    def create_network(self):
        self.layers = []
        for i in range(len(self.layer_size - 1)):
            self.layers.append(dense(self.layer_size[i], self.layer_size[i + 1]))

    
    def train(self):
        for e in range(self.epochs):
            pass

    def predict(self):
        pass

