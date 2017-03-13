import numpy as np

import cv2

IMAGE_SHAPE = (9, 6)

def collect_obj_img_pts(image_paths):
    """Collect object and image points from multiple images"""
    obj_pts = []
    img_pts = []

    # Prepare obj pts
    obj_pt = np.zeros((IMAGE_SHAPE[0] * IMAGE_SHAPE[1], 3), np.float32)
    obj_pt[:,:2] = np.mgrid[0:IMAGE_SHAPE[0], 0:IMAGE_SHAPE[1]].T.reshape(-1, 2)

    # Iterate through images
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, IMAGE_SHAPE, None)

        if ret == True:
            obj_pts.append(obj_pt)
            img_pts.append(corners)

    return obj_pts, img_pts


def calibrate_camera(obj_pts, img_pts, img_size):
    """Calibrate camera given object points, image points"""
    return cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)


def undistort(img, mtx, dist):
    """Undistort test image with matrix, distortion"""
    return cv2.undistort(img, mtx, dist, None, mtx)
