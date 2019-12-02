import numpy as np
import cv2
import math
import os

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
camFileNameLoad = 'camMatCircle03.npz'
dataTXT = 'test1.txt'


# til dataopsamling
pos = np.array([[], [], []], dtype=np.float64) # til at opsamle tvecs vektor
rot = np.array([[], [], []], dtype=np.float64) # til at opsamle rvecs vektor
unitVec = np.empty((3,0),dtype=np.float64)
pyr = np.empty((0,3), dtype=np.float64)

stdP = np.array([[0], [0], [0]], dtype=np.float64)
stdR = np.array([[0], [0], [0]], dtype=np.float64)
meanR = np.array([[0], [0], [0]], dtype=np.float64)
stdPyr = np.array([[0],[0],[0]], dtype=np.float64)

rotM = np.zeros((3,3))

calibrationCircleDimension = 0.02546 # circleGrid fra http://opencv.willowgarage.com/ printet til a4
patternSize = tuple((4,11)) #grid af indre hjoerner af checkerboard


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
def rotationMatrixToEulerAngles(R):
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

    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# laver ASym circle grid.
objp = []
for i in range(patternSize[1]):
    for j in range(patternSize[0]):
        objp.append(( (2*j + i%2)*calibrationCircleDimension/2,i*calibrationCircleDimension/2, 0) )
objp= np.array(objp).astype('float32')

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*calibrationCircleDimension

cap = cv2.VideoCapture(0) # 0 for intern webcam
cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret, img = cap.read() #læser data fra kamera
    #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) # rotere img hvis det er nødvendigt
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # gør img sort hvid, hjælper tracking ved at øge konstrast
    cret, centers = cv2.findCirclesGrid(gray, patternSize, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if cret:
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        cv2.drawChessboardCorners(img,patternSize,centers,cret)
        # Find the rotation and translation vectors.
        ret2 ,rvecs, tvecs, inliers,= cv2.solvePnPRansac(objp, centers, mtx, dist)
        cv2.Rodrigues(rvecs,rotM) # danner rotMatrix fra rotVektor
        # isRotationMatrix(rotM)
        eulerAng = np.array([rotationMatrixToEulerAngles(rotM)])
        print(eulerAng*180/math.pi)
        print(tvecs.T)
        #beregner unitvektor af rotationen - omdanner spheric coord til cartesian coord
        phi = eulerAng[0,2]    # yaw / azimuth
        theta = eulerAng[0,1] # pitch / inclination , men omvendt derfor defineres lambda
        lmbda = math.pi/2-theta # omdannet pitch
        
        uV = np.array([[ math.sin(lmbda) * math.cos(phi) ] , [math.sin(lmbda) * math.sin(phi)] , [-math.cos(lmbda)] ])
        # logger alle datapunkter
        unitVec = np.concatenate((unitVec, uV) , axis=1)
        pos = np.append(pos,tvecs, axis= 1)
        rot = np.append(rot,rvecs, axis= 1)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,centers,imgpts)
        
    
    
    cv2.imshow('Pose Estimation',img)
        
    if cv2.waitKey(1) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


for i in range(rot.shape[1]):
    cv2.Rodrigues(rot[:,i],rotM)
    isRotationMatrix(rotM)
    eulerAng = np.array([rotationMatrixToEulerAngles(rotM)])
    pyr = np.concatenate((pyr, eulerAng), axis = 0)

stdPyr = np.std(pyr, axis = 0, dtype=np.float64)
meanPyr = np.mean(pyr, axis = 0, dtype=np.float64)


for i in range(pos.shape[0]):
    #print(pos[i])
    #print(np.std(rot[i], axis = 0 , dtype=np.float64))
    stdP[i] = np.std(pos[i], axis = 0 , dtype=np.float64)
    stdR[i] = np.std(rot[i], axis = 0 , dtype=np.float64)
    #stdPyr[i] = np.std(pyr, axis = 1, dtype=np.float64)
    meanR[i] = np.mean(rot[i], axis = 0, dtype=np.float64)

print("Rotation in degrees (Roll, Pitch, Yaw)\n", meanPyr)
print("standard dev of Roll, Pitch, Yaw\n", stdPyr)
print("standard dev of Translation(position) in meters\n", stdP.T)

# gem til txt
if os.path.isfile(dataTXT):
    num = int(float(dataTXT[4:5])) 
    num += 1
    dataTXT = "test"+ str(num)+".txt"
    print("new filename",dataTXT)
np.savetxt(dataTXT,("stdPYR",stdPyr,"stdR",stdR.T,"stdP",stdP.T,"meanPYR",meanPyr,"numb. data points",pyr.shape[0],"Position",pos.T,"PitchYawRoll",pyr), fmt="%s")


fig = plt.figure()

ax1 = fig.add_subplot(1,3,1, projection='3d')
ax1.plot(pos[0], pos[2], -pos[1], label='positon')
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(-0.5, 0.5)
ax1.set_zlim(0, 1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(1,3,2, projection='3d')
ax2.quiver(0,0,0,unitVec[0],unitVec[2],-unitVec[1],label = 'rotation')
ax2.legend()
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.set_zlim(-1.1, 1.1)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax2 = fig.add_subplot(1,3,3)

ax2.scatter(objp[:,0], objp[:,1])


plt.show()
