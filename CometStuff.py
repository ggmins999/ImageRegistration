# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import pathlib

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

def loadComets(dir):
    for i in range(1,1000):
        path = f'{dir}/grid_a{i}.png'
        outpath = f'{dir}/comet_{i}.png'
        img = cv2.imread(path)
        if img is None:
            continue
        result = cometStats(img)
        cv2.imwrite(outpath,result)



k = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/comet1.png")

def sumPixels(img):
    np.sum(img[y1:y2, x1:xNN2, c1:c2])

def cometStats(comet, contour, w, h,n,outputdir):
    """maxX = 0
    maxVal = 0
    midy = int(h/2)
    for i in range(0,min(w,h)):
        pixel = comet[midy,i]
        if pixel >= maxVal:
            maxVal = pixel
            maxX = i

    maxX = int(h/2)
    """
    pathname = f'{outputdir}/comet{n}.png'
    cv2.imwrite(pathname,comet)
    #print(f"maxVal = {maxVal}")
    area = np.sum(comet)
    head = headMask(comet)
    diff = cv2.bitwise_and(comet,comet,mask = head)
    area2 = np.sum(diff)
    p = area2/area
    #print(w,h,area)
    return (w,h,area, area2,p)


    #cv2.circle(img,center=(maxX+x,midy+y),radius= 5,color= (0,0,255),thickness=2)
    #cv2.imshow("slice",slice)
    #cv2.imshow('contours', img)
    #cv2.waitKey(0)



def loadImage(path,name):
    outputdir = path[:-4]
    pathlib.Path(outputdir).mkdir(parents=True, exist_ok=True)
    start = time.time()
    im = cv2.imread(path)
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    totalw = im.shape[1]
    totalh = im.shape[0]
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 10 and  (w/h) > 1 and (w/h) < 8 and w< totalw/10:
            good_contours.append(contour)
    results = []
    n = 1
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        r = cometStats(gray[y:y+h,x:x+w], contour, w, h,n,outputdir)
        n = n+1
        results.append(r)
    end = time.time()
    print(f'loadImage took {end - start} msec')
    table = np.array(results)
    np.savetxt(f'{name}_stats.csv',table,delimiter = ",",fmt="%.2f",header="w,h,cometarea,totaltail,percent")
    #cv2.imshow('comets', img)
    cv2.waitKey(0)

def loadWells(path):
    obj = os.scandir(path)
    # List all files and diretories
    # in the specified path
    print("Files and Directories in '% s':" % path)
    for entry in obj:
        if entry.is_file() and entry.name.endswith('tif'):

            loadImage(entry.path,entry.name)

def headMask(c):
    width = c.shape[1]
    height = c.shape[0]
    head = c[0:height-1, 0:height-1]
    m = np.max(head)
    foo, t = cv2.threshold(c, m - 10, 255, cv2.THRESH_BINARY_INV)
    return(t)



#loadImage("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/Practicecomets.tif")
loadWells("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/PetersImages/TestData")



#loadComets("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images")


