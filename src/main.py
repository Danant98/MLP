#!/usr/bin/env python

__author__ = 'Daniel Elisabethsønn Antonsen, UiT Institute of statistics and mathematics'

# Importing libraries and modules
import scipy
import numpy as np
from util import ConfusionMatrix, Accuracy, Error, misslabeled

# Loading data 
data = scipy.io.loadmat('data/ExamData3D.mat')

# Loading training data
X_train, Y_train = data['X_train'], data['Y_train']

# Loading testing data
X_test, Y_test = data['X_test'], data['Y_test']


