# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:53:28 2023

@author: Abhi
"""

import cv2
from filters_ECE_481 import *
from skimage.io import imsave
from skimage.util import random_noise

inp_img = cv2.imread('C:/Users/Abhi/Downloads/IMG_0078.jpg')

img_gray = cv2.cvtColor(inp_img,cv2.COLOR_BGR2GRAY)

noisy = random_noise(img_gray, mode = 'salt')

f1 = am_filt(3, 3, noisy)

f2 = hm_filt(3, 3, noisy)

f3 = gm_filt(3, 3, noisy)

f4 = chm_filt(3, 3, noisy, -0.5)

imsave('AM.png',f1)
imsave('GM.png',f2)
imsave('HM.png',f3)
imsave('CH1.png',f4)