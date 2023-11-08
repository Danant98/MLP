#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np

class dense:

    def __init__(self, inpt_size:tuple, output_size:tuple):
        self.inpt = None
        self.output = None
        # Initializing weights and biases
        self.w = np.random.randn(output_size, inpt_size)
        self.b = np.random.randn(output_size, 1)
    
    def forward(self, inpt:np.ndarray):
        """
        Forward-propagation
        """
        self.inpt = inpt
        return np.dot(self.w, self.inpt) + self.b


    def backward(self, output_grad:np.ndarray, lr:float, momentum:float):
        """
        Back-propagation using gradient decsent with momentum
        """
        # Computing the gradient for the weights
        w_grad = np.dot(output_grad, self.inpt.T)
        inpt_grad = np.dot(self.w.T, output_grad)

        # Updating weights and biases
        self.w += momentum * self.w - lr * w_grad
        self.b += momentum * self.b - lr * inpt_grad





