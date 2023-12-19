#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np

class dense:

    def __init__(self, inpt_size:int, output_size:int):
        self.inpt = None
        self.output = None
        # Initializing weights and bias
        self.w = np.random.randn(inpt_size + 1, output_size)

    def activation(self, inpt:np.ndarray):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-inpt))
    
    def forward(self, inpt:np.ndarray):
        """
        Forward-propagation
        """
        # Stack ones in the first column to account for bias
        self.inpt = np.hstack(np.ones((inpt.shape[1], 1)), inpt)
        comp = np.dot(self.w, self.inpt)
        self.output = self.activation(comp)
        return self.output

    def backward(self, old_layer_delta:np.ndarray, layer_delta:np.ndarray, lr:float, momentum:float):
        """
        Back-propagation using gradient descent with momentum
        """
        # Computing the gradient for the weights
        w_grad = np.dot()
        # Updating weights and biases
        self.w += momentum * old_layer_delta - lr * w_grad




