#import the necessary packages
import numpy as np
import imutils
import cv2
import os
import pathlib
import matplotlib.pyplot as plt




def s(im=None):
    global k
    if im is not None:
        cv2.imshow('foo', im)
    else:
        cv2.imshow('foo', k)
    cv2.waitKey(0)

def sumPixels(img):
    np.sum(img[y1:y2, x1:xNN2, c1:c2])

def writeComet(comet,n,outputdir):
    pathname = f'{outputdir}/comet{n}.png' #writes number of comet out
    cv2.imwrite(pathname,comet)



def loadImage(path,name):
    outputdir = path[:-4]
    pathlib.Path(outputdir).mkdir(parents=True, exist_ok=True)
    #start = time.time()
    im = cv2.imread(path)
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #makes a greyscale copy of the image
    totalw = im.shape[1] #gets width of image
    totalh = im.shape[0] #gets heigh of image
    centerx = totalw//2
    centery = totalh//2
    thresh, bin = cv2.threshold(gray[centery-250:centery+250,centerx-250:centerx+250], 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, bin = cv2.threshold(gray, int(thresh*0.25), 255, cv2.THRESH_BINARY)#uses a fixed threshold taking everything brighter than 20 pixels is a part of the foreground
    print(thresh)
    #cv2.imshow("window",bin)
    #cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find contours
    good_contours = []
    bad_contours = []
    dilation = 1.2
    for contour in contours:#loops over all the contours and sorts what we have specified to be a comet
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        dilatedwidth = w*dilation
        dilatedheight = h*dilation
        dw = dilatedwidth-w
        dh = dilatedheight-h
        x = int(x-dw/2)
        y = int(y-dh/2)
        w = int(dilatedwidth)
        h = int(dilatedheight)
        if w < 5 or h < 5:
            continue
        mycomet = bin[y:y+h,x:x+w]
        if area > 25 and (w/h) > 0.5 and (w/h) < 8 and w< totalw/10 and area < 500000 and w>1 and h>1: # finds canditates specifies bounding box, has to be square or rectangular (accounts for low damage comets)
            good_contours.append(contour)
            print("good_contours",x,y,w,h,area)
        else:
            print("bad",w/h,area)
            if area < 50000:
                bad_contours.append(contour)
   # cv2.drawContours(img,bad_contours,-1,(0,0,255),3)
   # cv2.drawContours(img,good_contours,-1,(255,0,0),3)
   # plt.imshow(img)
    #plt.show()
       # cv2.imshow("window", mycomet)
    #cv2.waitKey(0)
    print(f'number of contours found = {len(good_contours)}')
    results = []
    #if len(good_contours] > 600:
        #end
   # else:
       # continue
    n = 1
    total = cv2.drawContours(img.copy(),good_contours,-1,(0,0,255),3)
    cv2.imwrite("total.png",total)
    for c in good_contours: #for candidates calls cometstats which are the equations to find percent damage, length , width etc
        x, y, w, h = cv2.boundingRect(c)
        #img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)e
        try:
            writeComet(gray[y:y+h,x:x+w], n,outputdir)
            n = n+1
        except ValueError as error:
            print("skipping",error)
            continue
#    print(f'loadImage took {end - start} msec')
    table = np.array(results)
    np.savetxt(f'{name}_stats.csv',table,delimiter = ",",fmt="%.2f",header="w,h,cometarea,totaltail,percent")
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








loadWells("/Users/gigiminsky/Downloads/IC001/Post-comet")







