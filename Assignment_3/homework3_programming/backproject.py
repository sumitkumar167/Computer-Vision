"""
CS 6384 Homework 1 Programming
Backprojection
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# backprojecting a depth image to a point cloud in the camera coordinate frame
# input: depth with shape (H, W)
# input: intrinsic_matrix, a 3x3 matrix
# output: a point cloud, pcloud with shape (H, W, 3)
#TODO: implement this function
def backproject(depth, intrinsic_matrix):
    H, W = depth.shape
    # initialize the point cloud array
    pcloud = np.zeros((H, W, 3), dtype=np.float32)

    # compute inverse of intrinsic matrix
    K_inv = np.linalg.inv(intrinsic_matrix)

    # loop over each pixel
    for y in range(H):
        for x in range(W):
            # homogenous pixel coordinates
            pixel = np.array([x, y, 1.0], dtype=np.float32)

            # apply inverse intrinsics
            ray = K_inv @ pixel

            #scale by depth to get 3D point
            pcloud[y, x, :] = depth[y,x] * ray

    return pcloud

