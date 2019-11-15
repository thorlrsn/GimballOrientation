import numpy as np
import cv2
from os import listdir, mkdir
from os.path import isfile, join



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:17:15 2019

@author: JHH
"""

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 15)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 15)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 15)
    return img



# Load previously saved data
with np.load('calibCamMat.npz') as X:
    mtx, dist, _,_ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
calibrationSquareDimension = 0.13864/6  #meter, pattern i fuld skaerm på 15''
calibrationSquareDimension = 0.0685/5 # lille pattern på iphone
patternSize = tuple((4,3)) #antal indre hjoerner af checkerboard
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((patternSize[1]*patternSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)
objp *= calibrationSquareDimension

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*calibrationSquareDimension



cap = cv2.VideoCapture(0)
cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret , img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cret, corners = cv2.findChessboardCorners(gray, patternSize)
    if cret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret2 ,rvecs, tvecs, inliers,= cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
    
    
    cv2.imshow('Pose Estimation',img)
        
    if cv2.waitKey(45) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break