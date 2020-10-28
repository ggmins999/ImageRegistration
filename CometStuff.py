#import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import pathlib
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

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

"""def loadComets(dir):
    for i in range(1,1000):
        path = f'{dir}/grid_a{i}.png'
        outpath = f'{dir}/comet_{i}.png'
        img = cv2.imread(path)
        if img is None:
            continue
        result = cometStats(img)
        cv2.imwrite(outpath,result)"""



k = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/comet1.png")

def sumPixels(img):
    np.sum(img[y1:y2, x1:xNN2, c1:c2])

def cometStats(comet, contour, w, h,n,outputdir):
    maxX = 0
    maxVal = 0
    midy = int(h/2)
    THRESH = 20
    comet = np.where(comet < THRESH, 0, comet)
    for i in range(0,min(w,h)):
        pixel = comet[midy,i]
        if pixel >= maxVal:
            maxVal = pixel
            maxX = i

    maxX = int(h/2)

    pathname = f'{outputdir}/comet{n}.png'
    cv2.imwrite(pathname,comet)
    #print(f"maxVal = {maxVal}")
    cometarea = np.sum(comet)
    (x,y,w,h), headcontour, head = headMask(comet)
    diff = cv2.bitwise_and(comet,comet,mask = head)
    headarea = np.sum(diff)
    p = (cometarea-headarea)/cometarea
    c = cv2.cvtColor(comet,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(c,[headcontour],0,(255,0,0),2,8)
    ret, bin = cv2.threshold(comet, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(comet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(c,contours,-1,(0,255,0),2,8)
    plt.imshow(c)
    #print(w,h,area)
    return (w,h,cometarea, headarea,p)


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
    ret, bin = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 1000 and  (w/h) > 1 and (w/h) < 8 and w< totalw/10 and area < 50000:
            good_contours.append(contour)
    results = []
    n = 1
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        #img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        r = cometStats(gray[y:y+h,x:x+w], contour, w,h,n,outputdir)
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
        if entry.is_file():
            parts = entry.name.split(".")
            if len(parts) > 1 and parts[-1] in ['tif', 'tiff', 'jpg','png','jpeg']:
                loadImage(entry.path, entry.name)


def headMask(comet):
    """ returns (  (x,y,w,h), contour_of_head, mask)"""
    width = comet.shape[1]
    height = comet.shape[0]
    head = comet[0:height-1, 0:height-1]
    m = np.max(head)
    blur = cv2.GaussianBlur(head, (5, 5), 0)
    tval, thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f'tval={tval}')
    tval, thresh = cv2.threshold(blur,(tval+10),255,cv2.THRESH_BINARY)
    plt.imshow(thresh)
    kernel = np.ones((5, 5), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_dilation = cv2.dilate(thresh, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=4)
    img_expand = cv2.dilate(img_erosion, kernel, iterations=2)
    img_final = cv2.erode(img_expand,kernel,iterations=)


    plt.imshow(img_final)

    cnts = cv2.findContours(img_final.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # find largest contour
    if len(sorted_ctrs) == 0:
        raise ValueError("no contours found in headMask")
    headcontour = sorted_ctrs[0]


    cometarray = np.zeros(comet.shape, dtype=np.uint8)
    cv2.drawContours(cometarray, headcontour, 0, 255, -1, 8)

    #plt.imshow(dist_transform)
    return (cv2.boundingRect(headcontour), headcontour, cometarray)

def headMask2(c):
    """ returns (  (x,y,w,h), contour_of_head, mask)"""
    width = c.shape[1]
    height = c.shape[0]
    head = c[0:height-1, 0:height-1]
    m = np.max(head)
    foo, t = cv2.threshold(head, 20, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(t, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    #plt.imshow(sure_fg)
    #plt.imshow(t)
    #contours, hierarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts, hierarchy = cv2.findContours(sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    #find largest contour
    if len(sorted_ctrs) == 0:
        raise ValueError("no contours found in headMask")
    headcontour = sorted_ctrs[0]
    cometarray = np.zeros(c.shape,dtype=np.uint8)
    cv2.drawContours(cometarray,[headcontour],0,255,-1,8)
    # return its bounding rectangle
    #plt.imshow(dist_transform)
    return (cv2.boundingRect(headcontour), headcontour, cometarray)



#loadImage("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/Practicecomets.tif")
loadWells("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/PetersImages/onetiff")



#loadComets("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images")


"""import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
#p = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/PetersImages/TestData/D8_-2_1_1_Stitched[Comet23jan19 Etopo2.5_GFP 469,525]_001/comet13.png")
p = cv2.imread("/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/PetersImages/TestData/D8_-2_1_1_Stitched[Comet23jan19 Etopo2.5_GFP 469,525]_001/D8_-2_1_1_Stitched[Comet23jan19 Etopo2.5_GFP 469,525]_001.tif")


c = cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)

#cv2.imshow("window",p)
#cv2.waitKey(0)

import numpy as np
max = np.max(c)
255
foo,t = cv2.threshold(c, max-10,255,cv2.THRESH_BINARY)


#plt.hist(c.ravel(),256,[0,256]);  plt.imshow(c); plt.show()


fig, (ax1, ax2) = plt.subplots(2)
ax1.hist(c.ravel(),256,[0,256])

c2 = c.copy()
(x,y,w,h), contour, head_mask = headMask(c)

cv2.rectangle(c2, (x, y), (x + w, y + h), (0, 0, 255), 1)

c2[0:5,0:10]=0
c2[c2<50]=0
ax2.imshow(c2)
plt.show()

k=input("press close to exit")"""