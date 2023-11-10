#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import numpy as np


def ConfusionMatrix(y_pred:np.ndarray, y:np.ndarray):
    """
    The confusion matrix for a classification problem
    """
    # Unique classes
    labels = np.unique(y)
    # Initializing confusion matrix
    conf_mat = np.zeros((labels.shape[0], labels.shape[0]))
    # Computing the confusion matrix
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            conf_mat[i, j] = np.sum((y_pred == labels[i]) & (y == labels[j]))
    return conf_mat

def Accuracy(conf_mat:np.ndarray, label:str = None):
    """
    Computing the accuracy of a given confusion matrix
    """
    accuracy = np.matrix.trace(conf_mat) / np.sum(conf_mat)
    if label is None:
        return accuracy
    return f'The accuracy for {label} is {accuracy * 100:.2f}%'

def Error(conf_mat:np.ndarray, label:str = None):
    """
    Computing the error of a given confusion matrix
    """
    error = 1 - Accuracy(conf_mat)
    if label is None:
        return error
    return f'The error for {label} is {error * 100:.2f}%'

def misslabeled(conf_mat:np.ndarray, label:str = None):
    """
    Computing the number of misslabeled samples
    """
    miss = np.sum(conf_mat) - np.matrix.trace(conf_mat)
    if label is None:
        return miss
    return f'The number of misslabeled samples for {label} is {miss}'



if __name__ == '__main__':
    pass 


