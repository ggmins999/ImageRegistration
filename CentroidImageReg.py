
import numpy as np
import time
import icp
import cv2
import imutils


# import matplotlib.pyplt as plt




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
    cX, cY = 0, 0
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(imgcopy, (cX, cY), 10, color, -1)
            cv2.putText(imgcopy, label, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("centroid.jpg", imgcopy)
    return ( cX, cY, imgcopy)



#    load_images('/Users/gigiminsky/Google Drive/B7Before/B7before.tif',
#    # '/Users/gigiminsky/Google Drive/B7After/B7after.tif')

def find_offset(pathb, patha):
    beforeimg = cv2.imread(pathb)
    afterimg = cv2.imread(patha)
    minwidth = int(min(beforeimg.shape[1], afterimg.shape[1]))
    minheight = int(min(beforeimg.shape[0], afterimg.shape[0]))
    beforeimg = beforeimg[0:minheight, 0:minwidth]
    afterimg = afterimg[0:minheight, 0:minwidth]

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


find_offset('/Users/gigiminsky/Google Drive/B4Before/B4before.tif',
            '/Users/gigiminsky/Google Drive/B4After/B4after.tif')