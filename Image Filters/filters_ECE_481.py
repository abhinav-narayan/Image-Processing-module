# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:10:12 2023

ECE 481 - Image Processing filters implementation

Arithmetic Mean Filter
Geometric Mean Filter
Harmonic Mean Filter
Wiener Filter
Contra Harmonic Mean Filter

@author: Abhi
"""

import cv2
import numpy as np

def am_filt(m,n,img):
    k = (1/m*n) * np.ones((m,n),dtype = 'uint8')
    f = cv2.filter2D(img,-1,k)
    return f

def gm_filt(m,n,img):
    img = np.float32(img)
    k = np.ones((m,n),dtype = 'uint8')
    f = np.exp(cv2.filter2D(np.log(img),-1,k))**(1/(m*n))
    return f

def hm_filt(m,n,img):
    img = np.float32(img)
    k = np.ones((m,n),dtype = 'uint8')
    d = cv2.filter2D(1/(img),-1,k)
    f = (m*n)/d
    return f

def chm_filt(m,n,img,q):
    img = np.float32(img)
    k = np.ones((m,n),dtype = 'uint8')
    f = cv2.filter2D(img**(q+1),-1,k)
    d = cv2.filter2D(f**q,-1,k)
    f = f/d
    return f

