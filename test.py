import numpy as np
import itertools
from IPython.display import display, Markdown, Latex, Math
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tropical

temp = np.load('saveFile.npz')
A1 = temp['A1']
A2 = temp['A2']
b1 = temp['b1']
b2 = temp['b2']

scale = 0

plt.rcParams['figure.figsize'] = [10, 12]
fig = plt.figure()
axF = fig.add_subplot(221, projection='3d')
axG = fig.add_subplot(222, projection='3d')
#fig.subplots_adjust(bottom = 1)

from matplotlib.widgets import TextBox

gray = '#59595B'
slate = '#105456'
def make_plot():
    A1temp = A1
    A2temp = A2
    b1temp = b1
    b2temp = b2
    if scale != 0:
        A1temp = scale*A1temp
        A2temp = scale*A2temp
        b1temp = scale*b1temp
        b2temp = scale*b2temp
        A1temp = np.round(A1temp).astype(int)
        A2temp = np.round(A2temp).astype(int)
        b1temp = np.round(b1temp).astype(int)
        b2temp = np.round(b2temp).astype(int)
    axF.clear()
    axG.clear()
    Fterms, Gterms = tropical.getTropCoeffs(A1temp, b1temp, A2temp, b2temp)
    tropical.newtonPolygon(Fterms, axF)
    tropical.newtonPolygon(Gterms, axG)
    axF.set_title('$F$')
    axG.set_title('$G$')

text_boxes = []
for j in range(A1.shape[0])[::-1]:
    for i in range(A1.shape[1]):
        def submit(text):
            A1[j, i] = float(text)
            #make_plot()
            #ax.draw()
        axbox = plt.axes([0.1 + 0.15*i, 0.3 - 0.05*(1-j), 0.075, 0.03])
        text_box = TextBox(axbox, '$A^{(1)}_{'+str(j+1) + str(i+1) + '}$', initial = str(A1[j, i]))
        text_box.on_submit(submit)
        text_boxes.append(text_box)   
        
for j in range(b1.shape[0])[::-1]:
    def submit(text):
        b1[j] = float(text)
        #make_plot()
        #ax.draw()
    axbox = plt.axes([0.5, 0.3 - 0.05*(1-j), 0.075, 0.03])
    text_box = TextBox(axbox, '$b^{(1)}_{'+str(j+1) + '}$', initial = str(b1[j]))
    text_box.on_submit(submit)
    text_boxes.append(text_box)   
    
for j in range(A2.shape[0])[::-1]:
    for i in range(A2.shape[1]):
        def submit(text):
            A2[j, i] = float(text)
            #make_plot()
            #ax.draw()
        axbox = plt.axes([0.1 + 0.15*i, 0.2 - 0.05*(1-j), 0.075, 0.03])
        text_box = TextBox(axbox, '$A^{(1)}_{'+str(j+1) + str(i+1) + '}$', initial = str(A2[j, i]))
        text_box.on_submit(submit)
        text_boxes.append(text_box)
        
for j in range(b2.shape[0])[::-1]:
    def submit(text):
        b2[j] = float(text)
        make_plot()
        #axF.draw()
        #axG.draw()
    axbox = plt.axes([0.7, 0.4 - 0.05*(1-j), 0.075, 0.03])
    text_box = TextBox(axbox, '$b^{(2)}_{'+str(j+1) + '}$', initial = str(b2[j]))
    text_box.on_submit(submit)
    text_boxes.append(text_box)   
    
def submit(text):
    global scale
    scale = float(text)
    print(scale)
    make_plot()
axbox = plt.axes([0.7, 0.3 - 0.05*(1-j), 0.075, 0.03])
text_box = TextBox(axbox, 'scale', initial = str(scale))
text_box.on_submit(submit)
text_boxes.append(text_box)   

make_plot()
plt.show()

