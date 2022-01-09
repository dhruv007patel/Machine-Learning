#!/usr/bin/python
# -*- coding: utf-8 -*-

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

targets = values[:, 1]
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Initialising empty dictationary train and test error

train_err = {}
test_err = {}

# Looping for different feature values

for feature in range(7, 15):

    x = values[:, feature]
    n = x.shape[0]
    ones = np.ones(n)
    mat_x = np.matrix(ones).reshape(n, 1)
    x = np.hstack((mat_x, x))

    # Applying Linear Regression without regularization, so lambda = 0 and polynomial feature = 3.

    (param, train_err[feature]) = a2.linear_regression(x=x[0:N_TRAIN],
            y=t_train, regLambda=0, order=3)

    # Passing the params and calculating error.

    test_err[feature] = a2.rms(x=a2.polynomial_features(x=x[N_TRAIN:],
                               order=3), y=t_test, w=param)

# Produce a plot of results.....

plt.bar(train_err.keys(), train_err.values(), 0.2)
plt.bar(np.add(list(test_err.keys()), 0.2), test_err.values(), 0.2)
plt.ylabel('RMS values')
plt.legend(['Training error', 'Test Error'])
plt.title('Model trained with degree 3 polynomial with no regularization')
plt.xlabel('Features')
plt.show()
