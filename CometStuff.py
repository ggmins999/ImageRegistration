# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os

import matplotlib.pyplot as plt
from math import  sqrt
#-8,7,61.5,35.25



def s(im=None):
    global k
    if im is not None:
        cv2.imshow('foo', im)
    else:
        cv2.imshow('foo', k)
    cv2.waitKey(0)

k = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/comet1.png")
k2 = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/comet2.png")
k3 = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/comet5.png")

def sumPixels(img):
    np.sum(img[y1:y2, x1:x2, c1:c2])

def cometStats(im, threshold):
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'found {len(contours)} contours')
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(contours[0])
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    slice = gray[y:y+h,x:x+w]
    maxX = 0
    maxVal = 0
    midy = int(h/2)
    for i in range(0,h):
        pixel = slice[midy,i]
        if pixel >= maxVal:
            maxVal = pixel
            maxX = i
    print(f"maxX = {maxX}")
    print(f"maxVal = {maxVal}")
    cv2.circle(img,center=(maxX+x,midy+y),radius= 5,color= (0,0,255),thickness=2)

    cv2.imshow("slice",slice)

    cv2.imshow('contours', img)
    cv2.waitKey(0)




cometStats(k3, 0)