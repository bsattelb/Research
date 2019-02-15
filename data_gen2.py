import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

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

def harderXor(n, r1 = 0.1, alpha=0.1):
    # Pick one of the four points
    # Pick which class
    # Pick an angle and a valid radius

    # r1 is inner circle radius
    r2 = (1+alpha)*r1  # r2 is annulus's inner radius
    r3 = np.sqrt(r1**2 + r2**2) # r3 is annulus's outer radius
    # Set r2 to be slightly larger than r1
    # Set r3 such that the inner circle and outer annulus have equivalent area

    data = np.zeros((n, 3))

    locs = [[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]] # The four group centers
    for i in range(n):
        locInd = np.random.randint(0, 4) # Which center?
        loc = locs[locInd]
        classif = np.random.choice([1, -1]) # Which class is it?
        angle = 2*np.pi*np.random.uniform() # What angle is this sample at?

        # What distance from the center is this sample at?
        if (np.sum(np.sign(loc)) == 0 and classif == 1) or (np.sum(np.sign(loc)) != 0 and classif == -1):
            r = (r3-r2)*np.random.uniform() + r2
        else:
            r = r1*np.random.uniform()

        # (r, theta) -> (x, y)
        x = loc[0] + np.cos(angle)*r
        y = loc[1] + np.sin(angle)*r

        # Set up data point
        data[i, :] = np.array([x, y, classif])

    return data, r2, r3

def harderXorVecOut(n, r1 = 0.1, alpha=0.1):
    # Pick one of the four points
    # Pick which class
    # Pick an angle and a valid radius

    # r1 is inner circle radius
    r2 = (1+alpha)*r1  # r2 is annulus's inner radius
    r3 = np.sqrt(r1**2 + r2**2) # r3 is annulus's outer radius
    # Set r2 to be slightly larger than r1
    # Set r3 such that the inner circle and outer annulus have equivalent area

    xVals = np.zeros((n, 2))

    yVals = np.zeros((n, 2))

    locs = [[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]] # The four group centers
    for i in range(n):
        locInd = np.random.randint(0, 4) # Which center?
        loc = locs[locInd]
        classif = np.random.choice([1, -1]) # Which class is it?
        angle = 2*np.pi*np.random.uniform() # What angle is this sample at?

        # What distance from the center is this sample at?
        if (np.sum(np.sign(loc)) == 0 and classif == 1) or (np.sum(np.sign(loc)) != 0 and classif == -1):
            r = (r3-r2)*np.random.uniform() + r2
        else:
            r = r1*np.random.uniform()

        # (r, theta) -> (x, y)
        x = loc[0] + np.cos(angle)*r
        y = loc[1] + np.sin(angle)*r

        # Set up data point
        xVals[i, :] = np.array([x, y])
        if classif == 1:
            yVals[i, :] = np.array([1, 0])
        else:
            yVals[i, :] = np.array([0, 1])

    return xVals, yVals

def plotClassifierData(inputVals, targetVals, plot=None):
    # Plot data
    xneg = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i, 0] > targetVals[i, 1]]
    yneg = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i, 0] > targetVals[i, 1]]

    xpos = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i, 0] < targetVals[i, 1]]
    ypos = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i, 0] < targetVals[i, 1]]

    if not plot:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    ax.scatter(xneg, yneg, s=1, color='b');
    ax.scatter(xpos, ypos, s=1, color='r');
    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1]);

def plotClassifierDataAlt(inputVals, targetVals, plot=None, alpha=1):
    # Plot data
    xneg = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i] < 0]
    yneg = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i] < 0]

    xpos = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i] > 0]
    ypos = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i] > 0]

    if not plot:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    ax.scatter(xneg, yneg, s=1, color='r', alpha=alpha);
    ax.scatter(xpos, ypos, s=1, color='g', alpha=alpha);
    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1]);
    
def plotClassifierDataShift(inputVals, targetVals, plot=None, alpha=1):
    # Plot data
    xneg = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i] < 0]
    yneg = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i] < 0]

    xpos = [np.asarray(inputVals[i, 0]) for i in range(targetVals.shape[0]) if targetVals[i] > 0]
    ypos = [np.asarray(inputVals[i, 1]) for i in range(targetVals.shape[0]) if targetVals[i] > 0]

    if not plot:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    ax.scatter(xneg, yneg, s=1, color='b', alpha=alpha);
    ax.scatter(xpos, ypos, s=1, color='r', alpha=alpha);
    ax.set_xlim([0, 2]);
    ax.set_ylim([0, 2]);

def regressionProblem(n, width, freq, power, cutoff):
    x = 2*np.random.rand(n, 1) - 1
    fx = np.abs(x)
    sinFx = (np.cos(freq*np.pi*fx) + 1)/2
    res = np.multiply(sinFx, np.exp(-power*np.maximum(0.0, np.power(fx, 2)-cutoff)))
    return x, res
