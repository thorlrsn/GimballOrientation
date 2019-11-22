import numpy as np
import cv2
from os import listdir, mkdir
from os.path import isfile, join



camFileNameSave = 'camMatCircle03'
maxSnap = 50
pattern = "AsCircle Grid" # 'Checkerboard' eller 'AsCircle Grid'.
# CB: afstand mellem intersect på checkerboard ELLER CG: horz/vert afstand mellem centrum af circler, ikke skrå/diagonal afstand.
calibrationDimension = 0.100/3 # circleGrid fra http://opencv.willowgarage.com/ printet til a4 .
patternSize = tuple((4,11)) #grid af indre hjoerner af checkerboard, eller antal rækker og "søjler" på circle grid.

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
      
objp = objPoints(pattern,calibrationDimension,patternSize)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

print("Detecting", pattern)
cap = cv2.VideoCapture(0) # 0 for intern webcam
snap = 0
#cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
while (cap.isOpened()):
    ret1, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Calibration", gray)
    if cv2.waitKey(1) & 0xff == ord(' '):
        # Find the chess board corners
        if pattern == "Checkerboard":
            ret, corners = cv2.findChessboardCorners(gray, patternSize)
            # If found, add object points, image points (after refining them)
        if pattern == "AsCircle Grid":
            ret, corners  = cv2.findCirclesGrid(gray, patternSize, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
            snap += 1
            cv2.imshow("Calibration", img)
            cv2.waitKey(500)

    if snap >= maxSnap: # gemmer calibrering når maxsnaps er opnået
        cv2.waitKey(1000)
        cap.release()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(camFileNameSave,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h) ,1 , (w,h))
        print("CALIBRATION DONE: Calibrationfile saved as" , camFileNameSave)
        break
    if cv2.waitKey(45) & 0xff == ord('q'): # stopper calibrering ved at trykke "q"
        cap.release()
        cv2.destroyAllWindows()
        print("CALIBRATION STOPPED, No calibrationfile saved.")
        break


# printer "calibreringsfejl"
tot_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    tot_error += cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    
mean_error = tot_error/len(objpoints)

print("total error of calibration: %f" % mean_error)
