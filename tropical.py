import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time


def getTropCoeffs(A1, b1, A2, b2, doTime=False):
    A1plus = np.maximum(A1, 0)
    A2plus = np.maximum(A2, 0)

    A1minus = -np.minimum(A1, 0)
    A2minus = -np.minimum(A2, 0)
    b1 = b1.reshape(-1, 1)

    Fterms = set({})
    Gterms = set({})

    Fbias = np.matmul(A2minus, A1minus)
    Fbias = np.append(Fbias, b2)

    Gbias = np.matmul(A2plus, A1minus)
    Gbias = np.append(Gbias, 0)
    
    if doTime:
        start = time.time()
        count = 0
        length = 2**np.size(A1, axis=0)

    for i in itertools.product([True, False], repeat=np.size(A1, axis=0)):
        B = np.zeros(A1.shape)
        B[i, :] = A1plus[i, :]
        B[tuple((not var for var in i)), :] = A1minus[tuple((not var for var in i)), :]

        c = np.zeros(b1.shape)
        c[i, :] = b1[i, :]

        B = np.concatenate((B, c), axis=1)

        Fterm = np.matmul(A2plus, B) + Fbias
        Gterm = np.matmul(A2minus, B) + Gbias
        Fterms.add(tuple(Fterm[0]))
        Gterms.add(tuple(Gterm[0]))
        
        if doTime:
            count = count + 1
            if count % 10000 == 0:
                print(str(100*count/length) + '% ' + str((time.time()-start)/(count/length)) + ' s remain')

    return Fterms, Gterms

def displayTropPoly(coeffs):
    s = ''
    addPlus = False

    for term in coeffs:
        if addPlus:
            s += ' \oplus '
        else:
            addPlus = True

        doConstant = True
        doX1 = True
        doX2 = True
        if term[2] == 0:
            doConstant = False
        if term[0] == 0:
            doX1 = False
        if term[1] == 0:
            doX2 = False

        s += '('

        if doConstant:
            s += '{term[2]}'.format(term=term)
        if doConstant and (doX1 or doX2):
            s += ' \odot '
        if doX1:
            s += 'x_1^{{\odot {term[0]}}}'.format(term=term)
        if doX1 and doX2:
            s += ' \odot '
        if doX2:
            s+= 'x_2^{{\odot {term[1]}}}'.format(term=term)
        if not doConstant and not doX1 and not doX2:
            s += '0'

        s += ')'
    return s

def newtonPolygon(coeffs, ax):
    ax.scatter([term[0] for term in coeffs], [term[1] for term in coeffs], [term[2] for term in coeffs]);
    ax.set_xlabel('$x_1$ degree')
    ax.set_ylabel('$x_2$ degree')
    ax.set_zlabel('Constant coefficient')
