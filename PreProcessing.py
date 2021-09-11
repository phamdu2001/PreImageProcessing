import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import feature as ft
import sklearn.externals
import pandas as pd

#show nhieu anh
def ShowImage(ImageList, nRows = 1, nCols = 2, WidthSpace = 0.0, HeightSpace = 0.00):
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(nRows, nCols)
    gs.update(wspace = WidthSpace, hspace = HeightSpace)
    plt.figure(figsize=(20,20))
    for i in range(len(ImageList)):
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        plt.subplot(nRows, nCols, n+1)

        image = ImageList[i].copy()
        if(len(image.shape)<3):
            plt.imshow(image, plt.cm.gray)
        else:
            plt.imshow(image)
        plt.title("Image" + str(i))
        plt.axis('off')

    plt.show()

#get list file in link
def get_subfiles(dir):
    checkPath = os.path.isdir(dir)
    print(checkPath)
    return next(os.walk(dir))[2]

#demo preprocess trafic
def preprocess_img(imgBGR, erode_dilate = True):
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLORS_BGR2HSV)

    Bmin = np.array([100, 42, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 42, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 42, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)

    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    imgbin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErode = np.ones((9,9), np.uint8)
        kernelDilate = np.ones((9,9), np.uint8)
        imgbin = cv2.erode(imgbin, kernelErode, iteration = 2)
        imgbin = cv2.dilate(imgbin, kernelDilate, iteration = 2)

    return imgbin

def coutour_detech(imgbin, min_area = 0, max_area = -1, wh_ratio = 2.0):
    rects =[]
    contours, _ = cv2.findContours(imgbin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = imgbin.shape[0]*img.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
                rects.append([x,y,w,h])
    return rects

def draw_rects_on_img(img, rects):
    img_copy = img.copy()
    for rect in rects:
        x,y,w,h = rect
        cv2.retangle(img_copy,(x,y),(x+w,y+h),(0,255,0), 3)
    return img_copy

