
import numpy as np
import os
import time
import cv2
import imutils
import pathlib
import matplotlib.pyplot as plt



def binarize(img):
    imgcopy = img.copy()
    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    # ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    otsu_threshold, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, )
    print("Obtained threshold: ", otsu_threshold)

    kernel = np.ones((5, 5), np.uint8)  # creates kernel for erosions and dilations
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_dilation = cv2.dilate(thresh, kernel, iterations=5)  # merges gaps 3 times
    thresh = cv2.erode(img_dilation, kernel, iterations=5)  # erodes back to original size 4 times
    return thresh

def find_centroid(img, label, color):
    imgcopy = img.copy()
    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    #ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    otsu_threshold, thresh = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,)
    print("Obtained threshold: ", otsu_threshold)

    kernel = np.ones((5, 5), np.uint8)  # creates kernel for erosions and dilations
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_dilation = cv2.dilate(thresh, kernel, iterations=5)  # merges gaps 3 times
    thresh = cv2.erode(img_dilation, kernel, iterations=5)  # erodes back to original size 4 times



    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    print(f' found {len(contours)} contours')
    cv2.drawContours(imgcopy,contours,-1,(255,0,0),3)
    #plt.imshow(imgcopy)
    sortedcontours= sorted(contours,key=lambda x: cv2.contourArea(x),reverse = True)
    print([ cv2.contourArea(c) for c in sortedcontours])
    assert len(contours) >0,'should just be 1 contour'
    cX, cY = 0, 0
    c = sortedcontours[0]
        # calculate moments for each contour
    M = cv2.moments(c)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(imgcopy, (cX, cY), 10, color, -1)
        cv2.putText(imgcopy, label, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("centroid.jpg", imgcopy)
    return ( cX, cY, imgcopy)

#def matchby_template(before,after):
    # find contours in the binary image
   # bbefore = binarize(before)
    #bafter = binarize(after)
   # result = cv2.matchTemplate(bafter,bbefore,cv2.TM_CCOEFF)
    #(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
    print(minLoc, maxLoc)



#    load_images('/Users/gigiminsky/Google Drive/B7Before/B7before.tif',
#    # '/Users/gigiminsky/Google Drive/B7After/B7after.tif')

def find_offset(beforeimg,afterimg):
    c1x, c1y, annotated_before = find_centroid(beforeimg, 'BEFORE', (255,255,255))
    cv2.imwrite('annotated_before.jpg', annotated_before)
    c2x, c2y, annotated_after = find_centroid(afterimg, 'AFTER', (255,255,255))
    cv2.imwrite('annotated_after.jpg', annotated_after)


    dx = c2x-c1x
    dy = c2y-c1y
    print(c1x,c1y,c2x,c2y)
    print(f'offset dx = {dx} , dy={dy}')
    translated = imutils.translate(annotated_before, dx, dy)
    translated[:, :, 0:2] = 0
    afterimg[:, :, 0] = 0
    afterimg[:, :, 2] = 0
    v = cv2.addWeighted(translated, 0.5, annotated_after, 0.5, 0)
    cv2.imwrite('overlay.jpg', v)
    return (dx,dy)
    #matchby_template(beforeimg,afterimg)


def find_correct_comet(pathb,patha,outputdir):
    beforeimg = cv2.imread(pathb)
    afterimg = cv2.imread(patha)
    minwidth = int(min(beforeimg.shape[1], afterimg.shape[1]))
    minheight = int(min(beforeimg.shape[0], afterimg.shape[0]))
    beforeimg = beforeimg[0:minheight, 0:minwidth]
    afterimg = afterimg[0:minheight, 0:minwidth]
    dx,dy = find_offset(beforeimg.copy(),afterimg.copy())

    pathlib.Path(outputdir).mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(afterimg.copy(), cv2.COLOR_BGR2GRAY)  # makes a greyscale copy of the image
    totalw = gray.shape[1]  # gets width of image
    totalh = gray.shape[0]  # gets heigh of image
    ret, bin = cv2.threshold(gray, 20, 255,
                             cv2.THRESH_BINARY)  # uses a fixed threshold taking everything brighter than 20 pixels is a part of the foreground
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    good_contours = []
    for contour in contours:  # loops over all the contours and sorts what we have specified to be a comet
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 100 and (w / h) > 0.25 and (
                w / h) < 8 and w < totalw / 10 and area < 50000:  # finds canditates specifies bounding box, has to be square or rectangular (accounts for low damage comets)
            good_contours.append(contour)
            # print(x,y,w,h,area)
    n = 0
    html = """<title>before/after slices</title>"""
    html += f'dx={dx},dy={dy}'
    html += """
            <table border=1> <th>point</th><th>before<th>after</tr>\n"""
    annotatedbefore = beforeimg.copy()
    annotatedafter = afterimg.copy()
    for c in good_contours:  # for candidates calls cometstats which are the equations to find percent damage, length , width etc
        n = n + 1
        x, y, w, h = cv2.boundingRect(c)
        aftercomet = gray[y:y + h, x:x + w]
        padding = 50
        beforecomet = beforeimg[y-dy-padding:y-dy + h+padding, x-dx-padding:x-dx + w+padding]
        afterfile = os.path.join(outputdir,f'comet{n}.png')
        beforefile = os.path.join(outputdir,f'beforecomet{n}.png')
        if beforecomet.shape[0] == 0 or beforecomet.shape[1] == 0:
            print(f'found zero dim beforecomet {n}  {beforecomet.shape}')
            continue
        aftercomet = cv2.cvtColor(aftercomet,cv2.COLOR_GRAY2BGR)
       # aftercomet[:,:,0] = 0
       # aftercomet[:,:,2] = 0
       ## beforecomet[:,:,0] = 0
       # beforecomet[:,:,1] = 0

        cv2.imwrite(afterfile,aftercomet)
        cv2.imwrite(beforefile,beforecomet)
        cv2.putText(annotatedbefore,f'{n}',(x-dx,y-dy),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(annotatedafter,f'{n}',(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        html = html + f'\n<tr><td>Point{n}: <td> <img src=beforecomet{n}.png> <td><img src=comet{n}.png></tr>\n'
        #print(n)

    html = html + '</table>'

    with open(os.path.join(outputdir,'index.html'), 'w') as out_file:
        out_file.write(html)
    cv2.imwrite(os.path.join(outputdir,"annotatebefore.png"),annotatedbefore)
    cv2.imwrite(os.path.join(outputdir, "annotateafter.png"), annotatedafter)

def match_images(bdir,adir,outputdir):
    afterfiles = os.scandir(adir)
    # List all files and diretories
    # in the specified path
    print("Files and Directories in '% s':" % adir)
    for afterentry in afterfiles:
        if afterentry.is_file():
            parts = afterentry.name.split(".")
            if len(parts) > 1 and parts[-1] in ['tif', 'tiff', 'jpg', 'png', 'jpeg']:
                filename = afterentry.name
                firsttwo = filename[0:2]
                bpath = find_file_prefix(bdir,firsttwo)
                resultdir = os.path.join(outputdir,os.path.splitext(filename)[0])
                print(resultdir)
                find_correct_comet(bpath,afterentry.path,resultdir)


def find_file_prefix(folder,prefix):
    entries = os.scandir(folder)
    for entry in entries:
        if entry.is_file() and entry.name[0:2] == prefix:
            return entry.path
    return None


#find_correct_comet('/Users/gigiminsky/Downloads/C10Before.tif',
                  #'/Users/gigiminsky/Downloads/C10After.tif',
                   #'/Users/gigiminsky/Google Drive/PyCharm Projects/C10Results')

match_images('/Users/gigiminsky/Google Drive/PyCharm Projects/BatchBeforeTest','/Users/gigiminsky/Google Drive/PyCharm Projects/BatchAfterTest',
             '/Users/gigiminsky/Google Drive/PyCharm Projects/TestResults')
