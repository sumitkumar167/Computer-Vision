"""
CS 6384 Homework 3 Programming
Epipolar Geometry
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


#TODO
# use your backproject function in homework 1, problem 2
from backproject import backproject
    
    
# read rgb, depth, mask and meta data from files
def read_data(file_index):

    # read the image in data
    # rgb image
    rgb_filename = 'data/%06d-color.jpg' % file_index
    im = cv2.imread(rgb_filename)
    
    # depth image
    depth_filename = 'data/%06d-depth.png' % file_index
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    depth = depth / 1000.0
    
    # read the mask image
    mask_filename = 'data/%06d-label-binary.png' % file_index
    mask = cv2.imread(mask_filename)
    mask = mask[:, :, 0]
    
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    # load matedata
    meta_filename = 'data/%06d-meta.mat' % file_index
    meta = scipy.io.loadmat(meta_filename)
    
    return im, depth, mask, meta
    
    
# Implement the 8-point algorithm to compute the fundamental matrix
# xy1 and xy2 are with shape (n, 2)
def compute_fundamental_matrix(xy1, xy2):
    """Compute fundamental matrix using 8-point algorithm.
    
    Args:
        xy1: Array of shape (n, 2) containing pixel coordinates in image 1
        xy2: Array of shape (n, 2) containing pixel coordinates in image 2
    
    Returns:
        F: 3x3 fundamental matrix
    """
    n = xy1.shape[0]
    
    # Normalize coordinates for numerical stability
    # Translate to origin
    mean1 = xy1.mean(axis=0)
    mean2 = xy2.mean(axis=0)
    xy1_norm = xy1 - mean1
    xy2_norm = xy2 - mean2
    
    # Scale so that mean distance from origin is sqrt(2)
    scale1 = np.sqrt(2) / np.mean(np.sqrt(xy1_norm[:, 0]**2 + xy1_norm[:, 1]**2))
    scale2 = np.sqrt(2) / np.mean(np.sqrt(xy2_norm[:, 0]**2 + xy2_norm[:, 1]**2))
    xy1_norm = xy1_norm * scale1
    xy2_norm = xy2_norm * scale2
    
    # Construct normalization matrices
    T1 = np.array([[scale1, 0, -scale1*mean1[0]],
                    [0, scale1, -scale1*mean1[1]],
                    [0, 0, 1]])
    T2 = np.array([[scale2, 0, -scale2*mean2[0]],
                    [0, scale2, -scale2*mean2[1]],
                    [0, 0, 1]])
    
    # step 1: construct the A matrix
    # For each correspondence, we have: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = xy1_norm[i, 0], xy1_norm[i, 1]
        x2, y2 = xy2_norm[i, 0], xy2_norm[i, 1]
        A[i, :] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # step 2: SVD of A
    # use numpy function for SVD
    U, S, Vt = np.linalg.svd(A)
    
    # step 3: get the last column of V
    # V is the transpose of Vt, so last column of V is last row of Vt
    F = Vt[-1, :].reshape((3, 3))
    
    # step 4: SVD of F
    U_f, S_f, Vt_f = np.linalg.svd(F)
    
    # step 5: mask the last element of singular value of F
    S_f[-1] = 0
    
    # step 6: reconstruct F
    F = np.matmul(U_f, np.matmul(np.diag(S_f), Vt_f))
    
    # Denormalize F: F = T2^T * F * T1
    F = np.matmul(T2.T, np.matmul(F, T1))

    return F  


# main function
if __name__ == '__main__':

    # read image 1
    im1, depth1, mask1, meta1 = read_data(6)
    
    # read image 2
    im2, depth2, mask2, meta2 = read_data(7)
    
    # intrinsic matrix
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
        
    # get the point cloud from image 1
    pcloud = backproject(depth1, intrinsic_matrix)
    
    # find the boundary of the mask 1
    boundary = np.where(mask1 > 0)
    x1 = np.min(boundary[1])
    x2 = np.max(boundary[1])
    y1 = np.min(boundary[0])
    y2 = np.max(boundary[0])
    
    # sample n pixels (x, y) inside the bounding box of the cracker box
    # due to the randomness here, you may not get the same figure as mine
    # this is fine as long as your result is correct    
    n = 10
    height = im1.shape[0]
    width = im1.shape[1]
    x = np.random.randint(x1, x2, n)
    y = np.random.randint(y1, y2, n)
    index = np.zeros((n, 2), dtype=np.int32)
    index[:, 0] = x
    index[:, 1] = y
    print(index, index.shape)

    # get the coordinates of the n pixels
    pc1 = np.ones((4, n), dtype=np.float32)
    for i in range(n):
        x = index[i, 0]
        y = index[i, 1]
        print(x, y)
        pc1[:3, i] = pcloud[y, x, :]
    print('pc1', pc1)
    
    # filter zero depth pixels
    ind = pc1[2, :] > 0
    pc1 = pc1[:, ind]
    index = index[ind]
    xy1 = index
    # xy1 is a set of pixels on image 1
    # we will find the correspondences of these pixels
    
    # transform the points to another camera
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    
    # Find correspondences: transform points from image 1 to image 2
    # ppc1 is in camera 1 frame, we need to transform to camera 2 frame
    # Step 1: Transform from camera 1 to world frame
    pc1_world = np.matmul(np.linalg.inv(RT1), pc1)
    
    # Step 2: Transform from world to camera 2 frame
    pc2 = np.matmul(RT2, pc1_world)
    
    # Step 3: Project to image 2 using intrinsic matrix
    # Use only first 3 rows of pc2 (XYZ coordinates, not homogeneous)
    xy2_homogeneous = np.matmul(intrinsic_matrix, pc2[:3, :])
    
    # Step 4: Convert to image coordinates (normalize by z)
    xy2 = np.zeros((pc2.shape[1], 2), dtype=np.int32)
    for i in range(pc2.shape[1]):
        if xy2_homogeneous[2, i] > 0:  # Check positive depth
            xy2[i, 0] = int(xy2_homogeneous[0, i] / xy2_homogeneous[2, i])
            xy2[i, 1] = int(xy2_homogeneous[1, i] / xy2_homogeneous[2, i])
    
    # Filter points that are within image bounds and have positive depth
    height2, width2 = im2.shape[:2]
    valid_idx = (xy2[:, 0] >= 0) & (xy2[:, 0] < width2) & (xy2[:, 1] >= 0) & (xy2[:, 1] < height2) & (pc2[2, :] > 0)
    xy1 = xy1[valid_idx]
    xy2 = xy2[valid_idx]
    
    # Implement the 8-point algorithm: compute fundamental matrix
    F = compute_fundamental_matrix(xy1, xy2)
    
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1 and sampled pixels
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: correspondences', fontsize=15)
    plt.scatter(x=xy1[:, 0], y=xy1[:, 1], c='y', s=20)
    
    # show RGB image 2 and sampled pixels
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: correspondences', fontsize=15)
    plt.scatter(x=xy2[:, 0], y=xy2[:, 1], c='g', s=20)
    
    # show three pixels on image 1
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: sampled pixels', fontsize=15)
    
    # compute epipolar lines of three sampled points
    px = 233
    py = 145
    p = np.array([px, py, 1]).reshape((3, 1))
    l1 = np.matmul(F, p)
    print(p.shape)
    print(l1) 
    plt.scatter(x=px, y=py, c='r', s=40)
    
    px = 240
    py = 245
    p = np.array([px, py, 1]).reshape((3, 1))
    l2 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='g', s=40)
    
    px = 326
    py = 268
    p = np.array([px, py, 1]).reshape((3, 1))
    l3 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='b', s=40)    
    
    # draw the epipolar lines of the three pixels
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: epipolar lines', fontsize=15)
    
    for x in range(width):
        y1 = (-l1[0] * x - l1[2]) / l1[1]
        if y1 > 0 and y1 < height-1:
            plt.scatter(x, y1, c='r', s=1)
            
        y2 = (-l2[0] * x - l2[2]) / l2[1]
        if y2 > 0 and y2 < height-1:
            plt.scatter(x, y2, c='g', s=1)
            
        y3 = (-l3[0] * x - l3[2]) / l3[1]
        if y3 > 0 and y3 < height-1:
            plt.scatter(x, y3, c='b', s=1)                        
                  
    plt.show()
