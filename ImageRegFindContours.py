import numpy as np
import time
import icp
import cv2
import imutils
#import matplotlib.pyplt as plt




def load_images(pathb,patha):
    beforeimg = cv2.imread(pathb)
    afterimg = cv2.imread(patha)
    minwidth = int(min(beforeimg.shape[1], afterimg.shape[1]))
    minheight = int(min(beforeimg.shape[0], afterimg.shape[0]))
    beforeimg = beforeimg[0:minheight, 0:minwidth]
    afterimg = afterimg[0:minheight, 0:minwidth]
    beforepoints = find_centers(beforeimg)
    afterpoints = find_centers(afterimg)
    T = test_icp(beforepoints,afterpoints)
    print(np.round(T*100)/100)
    overlay_imgs(beforeimg,afterimg,T)

def find_centers(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # makes a greyscale copy of the image
        ret, bin = cv2.threshold(gray, 20, 255,
                                 cv2.THRESH_BINARY)  # uses a fixed threshold taking everything brighter than 20 pixels is a part of the foreground
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours
        centers = []
        totalw = img.shape[1]
        for contour in contours:  # loops over all the contours and sorts what we have specified to be a comet
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area > 250 and (w / h) > 0.5 and (
                    w / h) < 8 and w < totalw / 10 and area < 50000:  # finds canditates specifies bounding box, has to be square or rectangular (accounts for low damage comets)
                    # compute the center of the contour
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append([cX,cY])
        print(f"n centers = {len(centers)}")
            # print(x,y,w,h,area)
        return np.array(centers)

def test_icp(B, A):


    blen = B.shape[0]
    alen = A.shape[0]
    if alen > blen:
        A = A[0:blen,:]
    elif blen > alen:
        B = B[0:alen,:]
    print(B.shape)
    print(A.shape)

    T, R1, t1 = icp.best_fit_transform(B, A)
    return T

def overlay_imgs(beforeimg,afterimg,T):
    shift1 = T[0,2]
    shift2 = T[1,2]
    rot = np.arccos(T[0,0])
    translated = imutils.translate(beforeimg, shift1, shift2)
    print(translated.shape)
    translated[:,:, 0:2] = 0
    afterimg[:,:,0] = 0
    afterimg[:,:,2] = 0
    v = cv2.addWeighted(translated,0.5,afterimg,0.5,0)
    cv2.imwrite('play.jpg',v)
    cv2.imshow("window",v)
    cv2.waitKey(0)


if __name__ == "__main__":
    load_images('/Users/gigiminsky/Google Drive/B7Before/B7before.tif','/Users/gigiminsky/Google Drive/B7After/B7after.tif')

