import cv2
import numpy as np
from PIL import Image,ImageFilter
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error
from skimage.util import random_noise

'''
Algorithms implemented:
Image Negative
Contrast Stretching
Image intensity transformation with and without background (With background - Image segmentation)
Histogram Equalization - cv2.equalizeHist(img_gray) 
Averaging - Mean filter - Without cv2.blur() function 
Median filter - cv2.medianBlur(img_gray,3) 
Min filter - PIL library 
Max filter - PIL Library 
'''

#Image Negative
def img_negative(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m,n = img_gray.shape
    new_img = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            new_img[i,j] = 255 - img_gray[i][j]
    return new_img

#Contrast Stretching
def contrast_stretching(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r0,s0 = 0,0
    r1,s1 = 20,50
    r2,s2 = 100,120
    r3,s3 = 255,255
    m,n = img_gray.shape
    new_img = np.zeros([m,n])
    alpha = (s1 - s0)/(r1 - r0)
    beta = (s2 - s1)/(r2 - r1)
    gamma = (s3 - s2)/(r3 - r2)
    for i in range(m):
        for j in range(n):
            if img_gray[i,j] <=0 and img_gray[i,j] < r1:
                new_img[i,j] = alpha * img_gray[i,j]
            elif img_gray[i,j] > r1 and img_gray[i,j] < r2:
                new_img[i,j] = (beta * (img_gray[i,j] - r1)) + s1
            elif img_gray[i,j] > r2 and img_gray[i,j] <=255:
                new_img[i,j] = gamma * (img_gray[i,j] - r2) + s2
    return new_img

#Intensity level slicing without background
def intensity_level_slicing(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m,n = img_gray.shape
    new_img = np.zeros([m,n])
    n0 = 50
    n1 = 100
    p = 75
    for i in range(m):
        for j in range(n):
            if img_gray[i,j] < n0 and img_gray[i,j] < n1:
                new_img[i,j] = 75
            else:
                new_img[i,j] = img_gray[i,j]
    return new_img

#Intensity level slicing with background - Image segmentation function without cv2.threshold()
def intensity_level_slicing_background(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    m,n = img_gray.shape
    new_img = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            if img_gray[i,j] < 127:
                new_img[i,j] = 0
            elif img_gray[i,j] > 127:
                new_img[i,j]  = 255
    return new_img




inp_img = cv2.imread('C:/Users/Abhi/OneDrive/Illinois Tech MS EE books/Digital Image Processing/Project/Tree.jpg')
img_PIL = Image.open('C:/Users/Abhi/OneDrive/Illinois Tech MS EE books/Digital Image Processing/Project/Tree.jpg').convert('L')

#Applying Min filter and Max filter

f1 = img_PIL.filter(ImageFilter.MinFilter(size=3))
f2 = img_PIL.filter(ImageFilter.MaxFilter(size=3))

#To convert PIL datatype to numpy array for OpenCV
f1_CV = np.array(f1)
f2_CV = np.array(f2)

cv2.imwrite('Min filter.png',f1_CV)
cv2.imwrite('Max filter.png',f2_CV)

img_gray = cv2.cvtColor(inp_img,cv2.COLOR_BGR2GRAY)

#Histogram Equalization
e = cv2.equalizeHist(img_gray)

o0 = img_negative(inp_img)
o1 = contrast_stretching(inp_img)
o2 = intensity_level_slicing(inp_img)


cv2.imwrite('Histogram_Equalization.png',e)
cv2.imwrite('Image negative.png',o0)
cv2.imwrite('Contrast stretching.png',o1)
cv2.imwrite('Intensity_level_slicing.png',o2)