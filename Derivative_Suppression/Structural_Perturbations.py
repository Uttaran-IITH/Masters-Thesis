#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
from skimage import transform as tr
from multiprocessing import Process, Queue, Manager, Value, Lock


# In[2]:


"""Loads MNIST and FMNIST data"""
def load(data):
    if data == "mnist":
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")    
    elif data == "fmnist":
        mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data,train_labels,eval_data,eval_labels


# In[6]:


"""Multi-Threaded Image transformation function.
    Local Variables :
    temp : Copy of data
    nprocs : Number of processes
    out_q : Output of image transformations for each process
    procs : Appends data from each process
    """
def Transform(name,eval_data,param):
    if name is "rotation" or name is "Rotation":
        name = Rotation
    elif name is "scaling" or name is "Scaling":
        name = Scaling
    elif name is "exposure" or name is "Exposure":
        name = Exposure
    elif name is "Shear" or name is "shear":
        name = Shear
    elif name is "Perspective" or name is "perspective":
        name = Perspective
    elif name is "Exposure" or name is "exposure":
        name = Exposure
    elif name is "Translation" or name is "translations":
        name = Translation
    temp = np.copy(eval_data)
    nprocs = 10
    out_q = Queue()
    procs = []
    for i in range(nprocs):
        p = Process(
                target=name,
                args=(temp,param,i+1,out_q,nprocs,temp.shape[0]))
        procs.append(p)
        p.start()
    resultdict = []
    for i in range(nprocs):
        resultdict.append(out_q.get())
    for p in procs:
        p.join()
    temp = np.array(resultdict).reshape(-1,785)
    return temp


# In[ ]:


"""Function for Horizontal Translation. l denotes the amount of translation"""
def Translation(eval_data,l,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    
    """Change M for vertical or 2d translation"""
    M = np.float32([[1,0,l],[0,1,0]])
    for i in range(0,eval_data.shape[0]):
        temp[i] = cv2.warpAffine(temp[i],M,(28,28))
        temp[i] = temp[i].clip(min=0,max=1)
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)


# In[2]:


"""Function for changing exposure. The output is clipped within the values of 0 and 1"""
def Exposure(eval_data,e,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    e = e/255
    temp+=e
    temp = np.clip(temp,0,1)
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)


# In[6]:


"""Function for rotating the image."""
def Rotation(eval_data,angle,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    s,r,c = temp.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),angle,1)
    for i in range (0,s):
        temp[i] = cv2.warpAffine(temp[i],M,(c,r))
        temp[i] = temp[i].clip(min=0,max=1)
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)


# In[1]:


"""Scales the given input. l must be >=0 """
def Scaling(eval_data,l,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    s,r,c = temp.shape
    for i in range (0,s):
        img = cv2.resize(temp[i],None,fx=l, fy=l, interpolation = cv2.INTER_CUBIC)
        if img.shape[0] > dim:
            m = np.int((img.shape[0]-dim)/2)
            temp[i] = img[m:m+dim,m:m+dim]
        else :
            m = dim
            img = cv2.copyMakeBorder( img, m, m, m, m, cv2.BORDER_CONSTANT)
            m = np.int((img.shape[0]-dim)/2)
            temp[i] = img[m:m+dim,m:m+dim]
            temp[i] = temp[i].clip(min=0,max=1)
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)


# In[ ]:


"""Changes the perspective of an image. l must be >=0 and an integer"""
def Perspective(eval_data,l,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    pts1 = np.float32([[0,0],[dim,0],[0,dim],[dim,dim]])
    pts2 = np.float32([[0,0],[l,0],[0,l],[l,l]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    for i in range(0,temp.shape[0]):
        temp[i] = cv2.warpPerspective(temp[i],M,(28,28))
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)


# In[ ]:


"""Perturbs the shear factor in transformed image. Change l accordingly"""
def Shear (eval_data,l,y,q,no_of_processes,size):
    m = 1
    x = 784
    n = 1
    dim = np.int(np.sqrt(x))
    p = (int)(size/no_of_processes)
    eval_data = eval_data[(y-1)*p:y*p]
    labels = eval_data[:,-1]
    eval_data = eval_data[:,0:784]
    temp = np.copy(eval_data.reshape(-1,dim,dim))
    afine_tf = tr.AffineTransform(shear=l)
    for i in range(0,temp.shape[0]):
        temp[i] = tr.warp(temp[i], inverse_map=afine_tf)
    temp = temp.reshape(-1,784)
    labels = labels.reshape(temp.shape[0],1)
    temp = np.hstack((temp,labels))
    q.put(temp)

