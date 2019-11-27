import numpy as np
import cv2
from os import listdir, mkdir
from os.path import isfile, join



camFileNameSave = 'camMat04'
maxSnap = 200

calibrationSquareDimension = 0.1330/5 # testgrid
patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.134/3 # mellemstort pattern i lab
#patternSize = tuple((8,5)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.0685/5 # lille pattern på iphone
#patternSize = tuple((4,3)) #grid af indre hjoerner af checkerboard

#calibrationSquareDimension = 0.13864/6  #meter, pattern i fuld skaerm på 15''
#patternSize = tuple((6,12)) #antal indre hjoerner af checkerboard


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((patternSize[1]*patternSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)
objp *= calibrationSquareDimension

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

print('Detecting checkerboard')
cap = cv2.VideoCapture(0) # 0 for intern webcam
snap = 0
#cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
while (cap.isOpened()):
    ret1, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Calibration', gray)
    if cv2.waitKey(1) & 0xff == ord(' '):
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, patternSize)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
            snap += 1
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)
    if snap >= maxSnap:
        cv2.waitKey(1000)
        cap.release()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(camFileNameSave,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h) ,1 , (w,h))
        break
    if cv2.waitKey(45) & 0xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

print('CALIBRATION DONE: saved as' , camFileNameSave)