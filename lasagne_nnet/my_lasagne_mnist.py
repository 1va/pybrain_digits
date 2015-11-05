"""
Description: Learning lasagne+theano on postcode digits dataset
Author: Iva
Date: Nov 2015
Python version: 2.7.10 (venv2)
"""

import sys
import os
import time

import numpy as np
import lasagne   #lightweigh nnet in theano
import theano
import theano.tensor as T
#from sklearn import cross_validation


def load_dataset():
    # open csv file (from kaggle)

    np_train = np.genfromtxt('data/train.csv', delimiter=',', skip_header= True, dtype='uint8')
    np_test = np.genfromtxt('data/test.csv', delimiter=',', skip_header= True, dtype='uint8')
    X = np_train[:,1:].reshape(42000,1,28,28)/np.float32(128)
    y = np_train[:,0]
    X_test = np_test.reshape(28000,1,28,28)/np.float32(128)
    y_test = np_train[:28000,0] # nonsense just to fit the shape

    sss = [(range(0,38000), range(38000,42000))]
        #cross_validation.StratifiedShuffleSplit(np_train[:,:1].ravel(), n_iter=1, test_size=.1, random_state=3476)
    for train_index, test_index in sss:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network



build_net = build_cnn

