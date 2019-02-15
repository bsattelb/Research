import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

def cosines(n, scale=0.8, shift=0.2):
    x = np.linspace(-1, 1, n)
    y1 = -scale*np.cos(1*math.pi*x) - shift
    y2 = -scale*np.cos(1*math.pi*x) + shift

    data1 = np.zeros((n, 3))
    data2 = np.zeros((n, 3))
    for i in range(n):
        data1[i, :] = np.array([x[i], y1[i], 1])
        data2[i, :] = np.array([x[i], y2[i], -1])

    dataset = np.vstack((data1, data2))

    return dataset, data1, data2

def circles(n):
    data1 = np.zeros((n, 3))
    data2 = np.zeros((n, 3))
    i = 0
    while i < n:
        sample = 2*np.random.uniform(size=(1,2)) - 1
        if np.sum(np.square(sample)) < (1/3)**2:
            data1[i, :] = np.append(sample, [1])
            i += 1

    i = 0
    while i < n:
        sample = 2*np.random.uniform(size=(1,2)) - 1
        if (2/3)**2 < np.sum(np.square(sample)) < 1:
            data2[i, :] = np.append(sample, [-1])
            i += 1

    dataset = np.vstack((data1, data2))

    return dataset, data1, data2

def plotData(data1, data2):
    fig, ax = plt.subplots()
    ax.plot(data1[:, 0], data1[:, 1], 'r.');
    ax.plot(data2[:, 0], data2[:, 1], 'b.');

def plotRegions(data1, data2, net, n=100, plot=None):
    predicted = []
    ymins = []
    n = 100
    for x1 in np.linspace(-1, 1, n):
        for y in np.linspace(-1, 1, n):
            pred = net.predict(np.array([x1, y]))
            predicted.append((x1, y, pred[0]))

    xmin = [pred[0] for pred in predicted if pred[-1]==-1]
    ymin = [pred[1] for pred in predicted if pred[-1]==-1]

    xmax = [pred[0] for pred in predicted if pred[-1]==1]
    ymax = [pred[1] for pred in predicted if pred[-1]==1]

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    ax.plot(xmin, ymin, 'ob', alpha=0.1)
    ax.plot(xmax, ymax, 'or', alpha=0.1)

    ax.plot(data1[:, 0], data1[:, 1], 'or');
    ax.plot(data2[:, 0], data2[:, 1], 'ob');
