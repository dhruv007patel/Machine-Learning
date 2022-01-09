#!/usr/bin/python
# -*- coding: utf-8 -*-

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

targets = values[:, 1]
x = values[:, 7:]
x = a2.normalize_data(x)

# Change feature value to get different plots as mentioned below
# feature = 3 : GNI
# feature = 4 : Life expectancy
# feature = 5 : Literacy

feature = 4

N_TRAIN = 100

# Select a single feature.

x_train = x[0:N_TRAIN, feature]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:, feature]
t_test = targets[N_TRAIN:]

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate

# np.asscalar(min(x_train)) -> min(x_train).item() : to fix the warning since function is
# deprecated with NumPy v1.16

x_ev = np.linspace(min(x_train).item(), max(x_train).item(),
                   num=500).reshape(500, 1)

# Applying Linear Regression without regularization, so lambda = 0 and polynomial feature = 3.

(param, tr_err) = a2.linear_regression(x=x_train, y=t_train,
        regLambda=0, order=3)

# Evaluate regression on the linspace samples.

y_ev = a2.polynomial_features(x=x_ev, order=3).dot(param)

# Produce a plot of results.....

plt.plot(x_ev, y_ev, 'r.-')
plt.plot(x_train, t_train, 'bo')
plt.plot(x_test, t_test, 'yo')
plt.ylabel('test_data_values')
plt.xlabel('train_data_values')
plt.title('A visualization of a regression estimate using random outputs'
          )
plt.show()
