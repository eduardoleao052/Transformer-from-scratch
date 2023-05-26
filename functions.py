import numpy as np
import pandas as pd
import scipy.signal

def add_padding(x,k):
    if k == 0:
        return x
    H,W = x.shape
    a = np.zeros((H+2*k,W+2*k))
    a[k:-k,k:-k] = x
    return a

def rotate_180(x):
    H,W = x.shape[0], x.shape[1]
    a = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            a[i, W-1-j] = x[H-1-i, j]
    return a
    
def cross_correlate(x,y,stride=1):
    H, W = x.shape
    HH, WW = y.shape
    Ho, Wo = 1 + (H - HH)//stride , 1 + (W - WW)//stride
    a = np.zeros((Ho,Wo))
    for i in range(Ho):
        for j in range(Wo):
            a[i,j] = np.sum(x[i*stride : HH + i*stride, j*stride : WW + j*stride] * y)
    return a

def convolute(x,y,stride=1):
    return cross_correlate(x,rotate_180(y),stride=stride)

x = np.array([[1,2,1],[2,0,3],[3,1,4]])
y = np.array([[1,2],[3,4]])


def softmax(z, training = False):
        z -= np.max(z,axis=0,keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z),axis=0,keepdims=True)
        return a

def sigmoid(z, training = False):
        #z= np.max(z,axis=1,keepdims=True)
        a = 1/(1+np.exp(-z))
        return a


#print(cross_correlate(x,y))
#print(scipy.signal.correlate2d(x,y,mode = 'valid'))
#x = add_padding(x,0)
#print(x)
#print(convolute(x,y))
#print(scipy.signal.convolve2d(x,y,mode = 'valid'))

