from CNN.forward import *
import numpy as np
import gzip
import os, glob
import cv2

def extract_data(directory, classes, data_type):
    y = None
    X = None
    split = 0.8
    for i, cls in enumerate(classes):
        print(directory+"/"+cls+"/*.tif")
        images = glob.glob(directory+"/"+cls+"/*.tif")
        split_length = int(len(images)*split)
        if data_type == 'train':
            images = images[:split_length]
        elif data_type == 'test':
            images = images[split_length:]
        for img in images:
            if y is None:
                y = np.array([i])
            else:
                y = np.append(y, i)
            if X is None:
                X = cv2.imread(img, -1)
                X = cv2.resize(X, (150, 150))
                X = X.reshape((1, X.shape[0]*X.shape[1]*X.shape[2]))
            else:
                new_x = cv2.imread(img, -1)
                new_x = cv2.resize(new_x, (150, 150))
                new_x = new_x.reshape((1, new_x.shape[0]*new_x.shape[1]*new_x.shape[2]))
                X = np.vstack((X,new_x))
    y = y.reshape([y.shape[0],1])
    return X, y

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation

    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity

    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)
