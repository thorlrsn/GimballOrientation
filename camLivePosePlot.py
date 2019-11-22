import numpy as np
import cv2
import math
import os

from os import listdir, mkdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation




# ---------------------------- funktioner ----------------------------

def objPoints (grid, cDimension, pSize):
    if grid == "Checkerboard":
        #prepare CB object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        op = np.zeros((pSize[1]*pSize[0],3), np.float32)
        op[:,:2] = np.mgrid[0:pSize[0],0:pSize[1]].T.reshape(-1,2)
        op *= cDimension
    elif grid == "AsCircle Grid":
        # prepare ASym circle grid object points.
        op = []
        for i in range(pSize[1]):
            for j in range(pSize[0]):
                op.append(( (2*j + i%2)*cDimension/2,i*cDimension/2, 0) )
        op= np.array(op).astype('float32')
    return op

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

def animate(i):
    ret, img = cap.read()
    # til dataopsamling
    pos = np.array([[], [], []], dtype=np.float64) # til at opsamle tvecs vektor
    rot = np.array([[], [], []], dtype=np.float64) # til at opsamle rvecs vektor
    unitVec = np.empty((3,0),dtype=np.float64)
    #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cret, corners = cv2.findChessboardCorners(gray, patternSize, cv2.CALIB_CB_FAST_CHECK)
    # Find the chess board corners
    if pattern == "Checkerboard":
        cret, corners = cv2.findChessboardCorners(gray, patternSize)
        # If found, add object points, image points (after refining them)
    elif pattern == "AsCircle Grid":
        cret, corners  = cv2.findCirclesGrid(gray, patternSize, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)
    if cret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret2 ,rvecs, tvecs, inliers,= cv2.solvePnPRansac(objp, corners2, mtx, dist)
        #danner rotm, rotationsmatrix
        rotM = np.empty((3,3))
        cv2.Rodrigues(rvecs,rotM)
        #isRotationMatrix(rotM)
        #beregner vinkler af roll pitch yaw
        eulerAng = np.array([rotationMatrixToEulerAngles(rotM)])
        print(eulerAng*180/math.pi) # printer vinkler til terminal
        #beregner unitvektor af rotationen - omdanner spheric coord til cartesian coord
        phi = eulerAng[0,2]    # yaw / azimuth
        theta = eulerAng[0,1] # pitch / inclination , men omvendt derfor defineres lambda
        lmbda = math.pi/2-theta # omdannet pitch
        
        uV = np.array([[ math.sin(lmbda) * math.cos(phi) ] , [math.sin(lmbda) * math.sin(phi)] , [-math.cos(lmbda)] ])
        # opsamler data
        unitVec = np.concatenate((unitVec, uV) , axis=1)
        pos = np.append(pos,tvecs, axis= 1)
        rot = np.append(rot,rvecs, axis= 1)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
    if cv2.waitKey(45) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        
    ax1.clear()
    # ax2.clear()
    ax1.scatter(0,0,0)
    ax1.quiver(pos[0], pos[2], -pos[1], 0.3*unitVec[0],0.3*unitVec[2],-0.3*unitVec[1], label='positon', linewidth = 3)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_zlim(-0.5, 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    # ax2.quiver(0,0,0,unitVec[0],unitVec[2],unitVec[1],label = 'rotation',linewidth=10)
    # ax2.set_xlim(-1.1, 1.1)
    # ax2.set_ylim(-1.1, 1.1)
    # ax2.set_zlim(-1.1, 1.1)
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Z')
    # ax2.set_zlabel('Y')

    cv2.imshow('Pose Estimation',img)


# -------------------- main -------------------------------


camFileNameLoad = 'camMatCircle03.npz'
pattern = "AsCircle Grid" # 'Checkerboard' eller 'AsCircle Grid'.
# CB: afstand mellem intersect på checkerboard ELLER CG: horz/vert afstand mellem centrum af circler, ikke skrå/diagonal afstand.
calibrationDimension = 0.100/3 # circleGrid fra http://opencv.willowgarage.com/ printet til a4 .
patternSize = tuple((4,11)) #grid a
# laver objp ud fra grid type.
objp = objPoints(pattern,calibrationDimension,patternSize)

# Checkerboarddata der søges efter
# calibrationSquareDimension = 0.13342/5 # test grid printet
# patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.0685/5 # lille pattern på iphone
#patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.13864/6  #meter, pattern i fuld skaerm på 15''
#patternSize = tuple((6,12)) #antal indre hjoerner af checkerboard

# Load previously saved data
with np.load(camFileNameLoad) as X:
    mtx, dist, _,_ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)*calibrationDimension

cap = cv2.VideoCapture(0) # 0 for intern webcam
cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection='3d')

# ax2 = fig.add_subplot(1,2,2, projection='3d')

print("Detecting", pattern)
#while(cap.isOpened()):
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

    
        #break