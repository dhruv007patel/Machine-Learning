#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Basic code for assignment 2."""

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.linalg import inv
from numpy.linalg import pinv


def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """

    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.

    data = pd.read_csv(fname, sep=',', na_values='_',
                       encoding='ISO-8859-1')

    # Strip countries title from feature names.

    features = (data.axes[1])[1:]

    # Separate country names from feature values.

    countries = data.values[:, 0]
    values = data.values[:, 1:]

    # Convert to numpy matrix for real.

    values = np.asmatrix(values, dtype='float64')

    # Modify NaN values (missing values).

    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """

    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec) / stdvec


# New helper function are added polynomial_features(),rms() and linear_regression()

def polynomial_features(x, order):
    n = x.shape[0]
    ones = np.ones(n)
    mat_x = np.matrix(ones).reshape(n, 1)
    x_ploy = np.hstack((mat_x, x))

    for i in range(2, order + 1):
        x_ploy = np.hstack((x_ploy, np.power(x, i)))

    return x_ploy


def rms(x, y, w):

    # y_pred = x * w

    y_bar = np.dot(x, w)

    # (y - y_pred)^2

    sq_pred = np.square(y - y_bar)

    # taking average

    mean_sq_pred = np.mean(sq_pred)

    # performing square root

    err = np.sqrt(mean_sq_pred)
    return err


def linear_regression(
    x,
    y,
    regLambda=0,
    order=0,
    ):

  # Calculating polynomial features on the input data : X

    x_ploy = polynomial_features(x, order)

  # Taking the transpose: X^T

    x_ploy_trans = x_ploy.transpose()

  # Multiplying: (X^T * X)

    x_ploy_squared = x_ploy_trans * x_ploy

    dim = x_ploy_squared.shape[1]

  # Condition check for regularization if regLambda != 0:
              # then optimal parameter (w) = [(X^T * X) + LI ]^(-1) * [(X^T)y]
              # else : (w) = [X^(-1) * y]

    if regLambda != 0:
        LI = regLambda * np.identity(dim)
        xTy = x_ploy_trans * y
        w = inv(x_ploy_squared + LI) * xTy
    else:
        w = pinv(x_ploy) * y

  # Calculating Loss

    l_rms = rms(x_ploy, y, w)
    return (w, l_rms)
