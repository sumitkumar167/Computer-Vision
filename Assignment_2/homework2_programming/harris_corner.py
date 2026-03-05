"""
CS 6384 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    # Apply a 3x3 max-filter over R (considers each pixel and its 8 neighbors).
    # A pixel is a local maximum if its value equals the neighborhood maximum AND
    # it is non-zero (i.e., it passed the threshold in step 6).
    local_max = maximum_filter(R, size=3)
    mask = np.zeros_like(R, dtype=np.uint8)
    mask[(R == local_max) & (R > 0)] = 1
    return mask


# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image
# Follow the steps in lecture_7_keypoint_features 1 slides 31-32
# You can use opencv functions and numpy functions
def harris_corner(im):

    # step 0: convert RGB to gray-scale image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    # ksize=3, ddepth=cv2.CV_64F so we keep negative gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # gradient in x direction
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # gradient in y direction

    # step 2: compute products of derivatives at every pixel
    Ix2  = Ix * Ix   # Ix^2
    Iy2  = Iy * Iy   # Iy^2
    Ixy  = Ix * Iy   # Ix * Iy

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    # GaussianBlur with a 5x5 kernel and sigmaX=sigmaY=0 (let OpenCV choose sigma from kernel size)
    Sx2  = cv2.GaussianBlur(Ix2, (5, 5), sigmaX=0)
    Sy2  = cv2.GaussianBlur(Iy2, (5, 5), sigmaX=0)
    Sxy  = cv2.GaussianBlur(Ixy, (5, 5), sigmaX=0)

    # step 4: compute determinant and trace of the M matrix
    # M = [[Sx2, Sxy],
    #      [Sxy, Sy2]]
    det_M   = Sx2 * Sy2 - Sxy * Sxy   # det(M)   = Sx2*Sy2 - Sxy^2
    trace_M = Sx2 + Sy2               # trace(M) = Sx2 + Sy2

    # step 5: compute R scores with k = 0.05
    k = 0.05
    R = det_M - k * (trace_M ** 2)

    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0

    # step 7: non-maximum suppression
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)
    
    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)
    
    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')
    
    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
