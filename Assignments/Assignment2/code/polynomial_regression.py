#!/usr/bin/python
# -*- coding: utf-8 -*-
import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()
targets = values[:, 1]
x = values[:, 7:]
x = a2.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Initialising empty dictationary to store train and test error

train_err = {}
test_err = {}

# Limiting order from 1-6. Was facing trouble with computation on laptop.

for order in range(1, 7):

# Applying Linear Regression without regularization, so lambda = 0 and different polynomial_feature = order.

    (param, train_err[order]) = a2.linear_regression(x=x_train,
            y=t_train, regLambda=0, order=order)

# Passing the params and calculating error.

    test_err[order] = a2.rms(x=a2.polynomial_features(x=x_test,
                             order=order), y=t_test, w=param)

# Produce a plot of results.

plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS values')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
