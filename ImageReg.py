# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os

import matplotlib.pyplot as plt
from math import  sqrt
#-8,7,61.5,35.25


# read args from command line
ap = argparse.ArgumentParser()

ap.add_argument("-b", "--before-image", help="Path to the 'before' image")
ap.add_argument("-a", "--after-image", help="Path to the 'after' image")
ap.add_argument("-p", "--params", help="Path to a previously saved grid params file")
ap.add_argument("-w", "--write-params", help="Path prefix to save  grid params")
ap.add_argument("-o", "--output", help="Path prefix to save output image files and annotations")

args = vars(ap.parse_args())

grid_params_file = args["params"] if args["params"] else 'grid-spacing.txt'
grid_param_outfile = args["write_params"] if args["write_params"] else 'grid-spacing.txt'
output_dir = args["output"] if args["output"] else 'images'
before_filename = args['before_image'] if args['before_image'] else "SYBR.jpg"
after_filename = args['after_image'] if args['after_image'] else "QDOT.jpg"


# check if output images directory exists, if not try to create it
os.makedirs(output_dir, exist_ok=True)

#loads in images
#before = cv2.imread("/Users/gigiminsky/Google Drive/Amelia Projects/B9_-1_1_1_Stitched[SYBR Comet_GFP 469,525]_001.jpg")
before_orig = cv2.imread(before_filename)
after_orig = cv2.imread(after_filename)

before = before_orig.copy()
after = after_orig.copy()

before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

ret, before = cv2.threshold(before, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, after = cv2.threshold(after, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# increment of rotation adjustment
r_angle = 0.25

#dir = "/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/"

#spacing = 20
height = before.shape[0]
width = before.shape[1]
cropleft = 50
cropright = 50
gridx = 246/4 #got this value from preview, divide by four to operate on resize image
gridy = 139/4 #got this value from preview, ^
resizedbefore = imutils.resize(before, width=int(width / 4.0))

# check if grid-spacing params file exists
if os.path.isfile(grid_params_file):
    with open(grid_params_file, 'r') as in_file:
      line = in_file.readline()
      (dx, dy, gridx, gridy, r_angle) = map(float, line.split(","))
else:
    (dx, dy, gridx, gridy, r_angle) = (-51.5,-63.0,60.25,33.75,0.25) # some default values

# we need to get some good guesses for these for the image
centerX = resizedbefore.shape[1] / 2
centerY= resizedbefore.shape[0]/2
wellRadius = resizedbefore.shape[0] / 2.5

# try it now

#def drawGrid(img,xoff,yoff): #function to draw the grid of circles, nested loops
    #w = img.shape[1]
    #h = img.shape[0]
    #for x in np.arange(cropleft, w-cropright, gridx):
     #   for y in np.arange(cropleft,h-cropright,gridy):
        #    p1 = (int(x+xoff),int(y+yoff))
         #   cv2.circle(img, p1, 10,(0,0,255),1) #draws circle grid across x axis
   # for x in np.arange(cropleft, w-cropright, gridx):
    #    for y in np.arange(cropleft,h-cropright,gridy):
    #        p1 = (int(x+xoff+gridx/2),int(y+yoff+gridy/2))
     #       cv2.circle(img, p1,10,(0,0,255),1) #draws circle grid across y axis
def drawGrid(img,xoff,yoff):
    for p1 in gridpoints(img, xoff, yoff, gridx, gridy):
        cv2.circle(img, p1, 10, (0, 0, 255), 1)  # draws circle grid across x axis

    with open('grid-spacing.txt', 'w') as out_file:
        out_file.write(f'{dx},{dy},{gridx},{gridy},{r_angle}')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'dx,dy,gridx,gridy,r_angle = {dx},{dy},{gridx},{gridy},angle={r_angle}', (0, 130), font, 1, (200, 255, 155), 2, cv2.LINE_AA)


def writeGridPointFile (img,radius,point,name):
    (x,y) = point
    shape = img.shape
    y1 = (y-radius)*4
    y2 = (y+radius)*4
    x1 = (x-radius)*4
    x2 = (x+radius)*4
    slice = img[y1:y2, x1:x2]
    if slice.size == 0:
        print(x,y, x1,x2,y1,y2, shape)
    #pathname = "/Users/gigiminsky/Google Drive/PyCharm Projects/ImageRegistration/Images/" + name + ".png"
    pathname =  name + ".png"
    print(f'wrote {pathname} x,y={x},{y}')
    cv2.imwrite(pathname,slice)
    slice = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(slice,10,100)
    #threshold = cv2.inRange(slice, 15, 255)
    return cv2.countNonZero(edges)




# let's add that grid point generator function
def gridpoints(img, xoff, yoff, gx, gy, center_x=centerX, center_y=centerY, max_radius=wellRadius):
    row = 0
    w = img.shape[1]
    h = img.shape[0]
    for y in np.arange(cropleft, h - cropright, gy / 2):
        for x in np.arange(cropleft, w - cropright, gx):
            # check if point (x,y) is within RADIUS distance of center, if
            # it is not, skip it
            distx = x - center_x
            disty = y - center_y
            dist = sqrt(distx*distx + disty*disty) # euclidean distance
            if dist > max_radius:
                continue     # this will skip this point by skipping to next iteration of the inner loop
            if row % 2 == 1:  # if row number is odd, shift the whole row horizontally by gridx/2
                shift = gx / 2
            else:
                shift = 0
            p1 = (int(x + xoff + shift), int(y + yoff))
            yield p1
        row = row + 1

def writeGrid(resized, imgbefore, imgafter,xoff,yoff, radius): #next i will take a slice out of this and add up the total intensities of the pixels get the sum of the array and write a loop that will try all the range of grid x grid y dx dy all the spacing and it will find the range that catches the most dots
    num = 1
    working = resized.copy()
    html = """<title>before/after slices</title>"""
    html += f'dx={dx},dy={dy},grid_x{gridx},grid_y{gridy}, r_angle={r_angle}'
    html += """
    <table border=1> <tr><th>grid point<th>before<th>after</tr>\n"""


    for p1 in gridpoints(resized, xoff, yoff, gridx, gridy):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x=p1[0]
        y=p1[1]

        cv2.rectangle(working, ((x-radius),(y-radius)), ((x+radius),(y+radius)), color=(200,220,100), thickness=2)
        cv2.putText(working, f'{num}', (p1[0]-10, p1[1]), font, 0.3, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("window", working)
        key = cv2.waitKey(1) & 0xFF
        # write slices from the before and after file
        pixels1 = writeGridPointFile( imgbefore, radius, p1, f"{output_dir}/grid_b{num}")
        pixels2 = writeGridPointFile( imgafter, radius, p1, f"{output_dir}/grid_a{num}")
        if pixels1 > 5:
            color = "green"
        elif pixels1 > 10:
            color = "dark green"
        else:
            color = "white"
        html = html + f'\n<tr><td bgcolor="{color}">Grid point {num}: {pixels1} {pixels2} <td> <img src=grid_b{num}.png> <td><img src=grid_a{num}.png></tr>'
        num = num + 1
    html = html + "</table>"

    with open(output_dir + '/slices.html', 'w') as out_file:
        out_file.write(html)
    cv2.waitKey(0) # wait until a key is hit to proceed

def countPixelsAtGridPoint (img,radius,point):
    (x,y) = point
    shape = img.shape
    y1 = (y-radius)
    y2 = (y+radius)
    x1 = (x-radius)
    x2 = (x+radius)
    # return sum of pixels in the ROI
    slice = img[y1:y2, x1:x2]
    edges = cv2.Canny(slice, 10, 100)
 #   vis = np.concatenate((slice, edges), axis=1)
 #   wname = f'slice{cv2.countNonZero(edges)}'
 #   cv2.imshow(wname, vis)
  #  cv2.waitKey(0)
  #  cv2.destroyWindow(wname)
    # threshold = cv2.inRange(slice, 15, 255)
    if cv2.countNonZero(edges) > 0:
        return 1
    else:
        return 0

def findGridResponse(img, xoff, yoff, gx, gy, angle, showdot=False):
    response = 0
    num = 1
    radius = 6
    w = img.shape[0]
    h = img.shape[1]
    img_rotated =  imutils.rotate(img.copy(), angle)


    for n, p1 in enumerate(gridpoints(img, xoff, yoff, gx, gy, max_radius=500)):
        if n % 20 != 0:
            continue
        dot = countPixelsAtGridPoint(img_rotated, radius, p1)
        response = response + dot
        if showdot

    return response


def findMaxGridResponse(img):
    max_response = 0
    max_params = (0,0,0,) # record what the params were when response was maximum
    #-8.0,6.0,61.5,35.25
    #-1.0,-47.0,62.0,35.0
    n = 0
    for xoff in np.arange(dx-1,dx+1,0.25):
        for yoff in np.arange(dy-1,dy+1,0.25):
            for gx in np.arange(gridx-1,gridx+1,0.25):
                for gy in np.arange(gridy-1,gridy+1,0.25):
                    resp = findGridResponse(img,xoff, yoff, gx, gy,angle)
                    n = n + 1
                    print(n,xoff,yoff,gx,gy,resp)
                    if resp > max_response:
                        max_response = resp
                        max_params = (xoff,yoff,gx, gy)
    print(f'max response was {max_response}, dx,dy,gridx,gridy={max_params}')
    return max_params


  #  num = 1
  #  radius = 20
  #  w = img.shape[1]/4
  #  h = img.shape[0]/4
  #  for x in np.arange(cropleft, w-cropright, gridx):
   #     for y in np.arange(cropleft,h-cropright,gridy):
   #         p1 = (int(x+xoff),int(y+yoff))
   #         writeGridPointFile(img,radius,p1,f"grid{num}")
    #        num = num+1
    #for x in np.arange(cropleft, w-cropright, gridx):
    #    for y in np.arange(cropleft,h-cropright,gridy):
      #      p1 = (int(x+xoff+gridx/2),int(y+yoff+gridy/2))
      #      writeGridPointFile(img,radius,p1,f"grid{num}")
      #      num = num+1

help = """
        r\toffset right
        l\toffset left
        u\toffset up
        d\toffset down
        +\tincrement gridx
        -\tdecrement gridx
        y\tincrement gridy
        t\tdecrement gridy
        q\tquit
        w\tWrite Grid images
        1\trotate right
        2\trotate left
        z\ttoggle grid
        """

writing = False
show_grid = True
while True:
    # make copy of the original images, rotate them by the current rotation angle
    before = imutils.rotate(before_orig.copy(), r_angle)
    after = imutils.rotate(after_orig.copy(), r_angle)
    resizedbefore = imutils.resize(before, width=int(width / 4.0))

    resizedbefore = cv2.cvtColor(resizedbefore, cv2.COLOR_BGR2GRAY)

    #ret, resizedbefore = cv2.threshold(resizedbefore, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, resizedbefore = cv2.threshold(resizedbefore, 20, 255, cv2.THRESH_BINARY)

    working = cv2.cvtColor(resizedbefore, cv2.COLOR_GRAY2BGR)

    if show_grid:
        drawGrid(working,dx,dy)
    cv2.imshow("window", working)
    key = cv2.waitKey(1) & 0xFF
    if chr(key) == 'r':
        dx = dx+1
    elif chr(key) == 'l':
        dx = dx-1
    elif chr(key) == 'u':
        dy = dy - 1
    elif chr(key) == 'd':
        dy = dy + 1
    elif chr(key) == '+':
        gridx = gridx+0.25
    elif chr(key) == '-':
        gridx = gridx-0.25
    elif chr(key) == 'y':
        gridy = gridy + 0.25
    elif chr(key) == 't':
        gridy = gridy - 0.25
    elif chr(key) == 'q':
        break
    elif chr(key) == 'w':
        writeGrid(working,before, after, dx,dy, radius=15)
    elif chr(key) == 'm':
        before_gray =  cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        #before_thresh = cv2.adaptiveThreshold(before_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        ret, before_thresh = cv2.threshold(before_gray,80,255,cv2.THRESH_BINARY)
        #kernel = np.ones((7, 7), np.uint8)
        #before_thresh = cv2.erode(before_thresh, kernel )
        before_thresh = imutils.resize(before_thresh, width=int(width / 4.0))

        dx,dy,gridx,gridy = findMaxGridResponse(before_thresh)
    elif chr(key) == '1':
        r_angle += 0.25
    elif chr(key) == '2':
        r_angle -= 0.25
    elif chr(key) == 'z':
        show_grid = not show_grid
    elif chr(key) == '?':
        print(help)

