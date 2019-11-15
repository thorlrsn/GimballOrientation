import numpy as np
import cv2
import math

from os import listdir, mkdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:17:15 2019

@author: JHH
"""
loadCalibMatrix = True
#makeCal = not loadCalibMatrix
#saveCamMat = False
camFileNameLoad = 'camMat01.npz'
#camFileNameSave = 'camMat02'
snap = 0
maxSnap = 30

pos = np.array([[], [], []], dtype=np.float64) # til at opsamle tvecs vektor
rot = np.array([[], [], []], dtype=np.float64) # til at opsamle rvecs vektor
pyr = np.array([[], [], []], dtype=np.float64)
#print(pos)
#rot
#pos = np.append(pos,[[1],[2],[3]], axis= 1)
#pos = np.append(pos,[[2],[3],[5]], axis= 1)
#pos = np.append(pos,[[1],[7],[3]], axis= 1)
stdP = np.array([[0], [0], [0]], dtype=np.float64)
stdR = np.array([[0], [0], [0]], dtype=np.float64)
meanR = np.array([[0], [0], [0]], dtype=np.float64)

rotM = np.zeros((3,3))


#print(std[1])
#print(mean)

calibrationSquareDimension = 0.134/3 # mellemstort pattern i lab
patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.0685/5 # lille pattern på iphone
#patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.13864/6  #meter, pattern i fuld skaerm på 15''
#patternSize = tuple((6,12)) #antal indre hjoerner af checkerboard

# funktioner ----

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 6)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 6)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 6)
    return img

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# main ----

# Load previously saved data
if loadCalibMatrix:
    with np.load(camFileNameLoad) as X:
         mtx, dist, _,_ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
# elif makeCal: # laver live calibrering
    
#     termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((patternSize[1]*patternSize[0],3), np.float32)
#     objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)
#     objp *= calibrationSquareDimension

#     Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.

#     print('Detecting checkerboard')
#     cap = cv2.VideoCapture(0) # 0 for intern webcam

#     cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)
#     cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
#     while (cap.isOpened()):
#         ret1, img = cap.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('Calibration', gray)
#         if cv2.waitKey(1) & 0xff == ord(' '):
#             Find the chess board corners
#             ret, corners = cv2.findChessboardCorners(gray, patternSize)
#             If found, add object points, image points (after refining them)
#             if ret == True:
#                 objpoints.append(objp)

#                 corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#                 imgpoints.append(corners2)

#                 Draw and display the corners
#                 img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
#                 snap += 1
#                 cv2.imshow('Calibration', img)
#                 cv2.waitKey(500)
#         if snap >= maxSnap:
#             cv2.waitKey(1000)
#             cap.release()
#             ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#             ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#             if saveCamMat: 
#                 np.savez(camFileNameSave,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
#                 print('calibration matrix saved')
#             h,  w = img.shape[:2]
#             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h) ,1 , (w,h))
#             break
#         if cv2.waitKey(45) & 0xff == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             break
#else:
    
    
    
    # for picam
    #foL = 0.00304
    # 1080p opløsning på video
    #x = 960
    #y = 540
    #fov 
    #fov_x = 62.2 # grader
    #fov_y = 48.8 # grader

    #f_x = x / np.tan(fov_x / 2)
    #f_y = y / np.tan(fov_y / 2)
    # camMat and dist
    #mtx = np.array([[f_x,0,x],[0,f_y,y],[0,0,1]])
    #dist = np.array([0.0,0,0,0,0])
     

    
#calibrationSquareDimension = 0.13864/6  #meter, pattern i fuld skaerm på 15''
#calibrationSquareDimension = 0.0685/5 # lille pattern på iphone
#patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((patternSize[1]*patternSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)
objp *= calibrationSquareDimension

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*calibrationSquareDimension

cap = cv2.VideoCapture(0) # 0 for intern webcam

cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret , img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cret, corners = cv2.findChessboardCorners(gray, patternSize)
    if cret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret2 ,rvecs, tvecs, inliers,= cv2.solvePnPRansac(objp, corners2, mtx, dist)
        #print(rvecs)
        #print(tvecs)
        pos = np.append(pos,tvecs, axis= 1)
        rot = np.append(rot,rvecs, axis= 1)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        
    
    
    cv2.imshow('Pose Estimation',img)
        
    if cv2.waitKey(45) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

#print(pos)
for i in range(pos.shape[0]):
    #print(pos[i])
    #print(np.std(rot[i], axis = 0 , dtype=np.float64))
    stdP[i] = np.std(pos[i], axis = 0 , dtype=np.float64)
    stdR[i] = np.std(rot[i], axis = 0 , dtype=np.float64)
    meanR[i] = np.mean(rot[i], axis = 0 , dtype=np.float64)
    
for i in range(rot.shape[1]):
    cv2.Rodrigues(rot[:,i],rotM)
    isRotationMatrix(rotM)
    eulerAng = rotationMatrixToEulerAngles(rotM)
    eulerAng2 = np.array([eulerAng])
    eulerAng2 = np.transpose(eulerAng2)
    print(eulerAng2)
    pyr = np.append(pyr, eulerAng2, axis= 1)

print('pyr' , pyr[0])


print("standard dev of tvecs" ,stdP)
print("standard dev of rvecs" ,stdR)

fig = plt.figure()

ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot(pos[0], pos[1], pos[2], label='positon')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.quiver(0,0,0,rot[0],rot[1],rot[2],label = 'rotaion')
ax.legend()
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-2.0, 2.0)

plt.show()
