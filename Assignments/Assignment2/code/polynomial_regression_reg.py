#!/usr/bin/python
# -*- coding: utf-8 -*-

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

# Defining different values lambda in form of list

lmb = [
    0,
    .01,
    0.1,
    1,
    10,
    100,
    1000,
    10000,
    ]

targets = values[:, 1]
x = values[:, 7:]
x = a2.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Initialising empty dictationary

lmb_average = {}

for l in lmb:

    # Initialising empty list

    te_error = []

    # one fold = 10 data points. So looping 10 times for 100 datapoints.

    for i in range(1, 11):

        # Partitioning data into train and test

        v = (i - 1) * 10
        u = i * 10
        x_tr_data = np.vstack((x_train[0:v, :], x_train[u:, :]))
        x_v_data = x_train[v:u, :]
        t_tr_data = np.vstack((t_train[0:v, :], t_train[u:, :]))
        t_v_data = t_train[v:u, :]

        # Applying Linear Regression with different lambda and polynomial feature = 2.

        (param, tr_err) = a2.linear_regression(x=x_tr_data,
                y=t_tr_data, regLambda=l, order=2)

        # Passing the params and calculating error.

        ts_err = a2.rms(x=a2.polynomial_features(x=x_v_data, order=2),
                        y=t_v_data, w=param)
        te_error.append(float(ts_err))

    # Storing average-error as per lambda value for plotting

    lmb_average[l] = np.mean(te_error)

# Produce a plot of results.....

plt.semilogx(lmb_average.keys(), lmb_average.values())
plt.ylabel('RMS values')
plt.xlabel('Regularization parameter - (lambda)')
plt.title('Polynomials fitted with regularization')
plt.show()
