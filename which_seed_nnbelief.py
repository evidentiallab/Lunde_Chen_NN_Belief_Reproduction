'''
Determine the seed of numpy.random, so as to Reproduce very similar
figures as in Thierry's paper.
Viable choices:
One pass: [4, 73, 101, 127, 149, 223, 234]
Two passes: (calling gen_data function twice)[4, 7, 128]
'''

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, Dense, Dropout, \
    Activation, ActivityRegularization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler

mu1 = [0, 0]
mu2 = [0, 0]
mu3 = [1, -1]
sigma1 = [[0.1, 0], [0, 0.1]]
sigma2 = [[0.5, 0], [0, 0.5]]
sigma3 = [[0.3, -0.15], [-0.15, 0.3]]
p1 = 1.0/3.0
p2 = 1.0/3.0
p3 = 1 - p1 - p2

def gen_data(N, mu1, mu2, mu3, sigma1, sigma2, sigma3, p1, p2):
    y = np.random.choice([1, 2, 3], N, p=[p1, p2, 1 - p1 - p2])
    X = np.zeros((N, 2))
    N1 = np.count_nonzero(y == 1)
    N2 = np.count_nonzero(y == 2)
    N3 = np.count_nonzero(y == 3)
    X[y==1, ] = np.random.multivariate_normal(mu1, sigma1, N1)
    X[y==2, ] = np.random.multivariate_normal(mu2, sigma2, N2)
    X[y==3, ] = np.random.multivariate_normal(mu3, sigma3, N3)
    return X, y

def get_predict_result(_px, _py):
    which = np.argmax(model.predict(
        scaler.transform([[_px, _py]]))[0])
    list_class_result_x[which].append(_px)
    list_class_result_y[which].append(_py)
    return which + 1

for i in range(2, 10000):
    print("i = " + str(i))
    np.random.seed(i)
    X, y = gen_data(900, mu1, mu2, mu3, sigma1, sigma2, sigma3, p1, p2)
    X, y = gen_data(900, mu1, mu2, mu3, sigma1, sigma2, sigma3, p1, p2)
    scaler = StandardScaler().fit(X)
    X_train = scaler.transform(X)
    y_train = to_categorical(y - 1, num_classes=3, dtype ="uint8")
    num_labels = len(np.unique(y))

    model = keras.Sequential(
        [
            InputLayer(2),
            Dense(20, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="relu", kernel_regularizer=regularizers.L2(0.5)),
            Dense(num_labels, activation="softmax"),
        ]
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=0)
    print("Train finished")
    feature_x = np.arange(-2.3, 2.3, 0.05)
    feature_y = np.arange(-2.3, 2.3, 0.05)
    [grid_X, grid_Y] = np.meshgrid(feature_x, feature_y)

    list_class_result_x = [[], [], []]
    list_class_result_y = [[], [], []]

    Z = np.vectorize(get_predict_result)(grid_X, grid_Y)
    print("vectorize finished")

    fig, ax = plt.subplots()

    ax.contour(grid_X, grid_Y, Z, levels=[1.5, 2.5, 3.5], colors='b')

    fig.set_size_inches(5, 4.3)
    plt.xlim([-2.3, 2.3])
    plt.ylim([-2.3, 2.3])
    plt.savefig(str(i) + '.png', bbox_inches='tight')