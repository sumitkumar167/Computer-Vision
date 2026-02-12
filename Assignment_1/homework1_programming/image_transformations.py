"""
CS 6384 Homework 1 Programming
Transform images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def inside_image(x, y, im):
    H = im.shape[0]
    W = im.shape[1]
    return x >= 0 and x < W and y >= 0 and y < H

# transform the input image im (H, W, 3) according to the 2D transformation T (3x3 matrix)
# the output is the transformed image with the same shape (H, W, 3)
#TODO: implementation this function
def transform(im, T):
    H = im.shape[0]
    W = im.shape[1]
    im_new = np.zeros_like(im, dtype=np.uint8)
    # Precompute inverse of T for efficiency
    T_inv = np.linalg.inv(T)
    # 1. Loop over each pixel in the output image
    for y_out in range(H):
        for x_out in range(W):
            # 2. Convert pixel to homogeneous coordinates
            p_out = np.array([x_out, y_out, 1], dtype=np.float32)
            # 3. Apply the inverse transformation to get the corresponding pixel in the input image
            p_in = T_inv @ p_out
            # Normalise homogeneous coordinates
            if abs(p_in[2]) < 1e-8:
                continue
            x_in = p_in[0] /p_in[2]
            y_in = p_in[1] /p_in[2]
            # 4. Check if the pixel is within the bounds of the input image
            if inside_image(x_in, y_in, im):
                # 5. If it is, copy the pixel value to the output image
                im_new[y_out, x_out] = bilinear_sample(im, x_in, y_in)                
    return im_new

# bilinear sampling
def bilinear_sample(im, x, y):
    H, W = im.shape[0], im.shape[1]
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    # out of bounds error --> return black
    if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
        return np.zeros((im.shape[2],), dtype=np.uint8)
    
    Ia = im[y0, x0].astype(np.float32)
    Ib = im[y0, x1].astype(np.float32)
    Ic = im[y1, x0].astype(np.float32)
    Id = im[y1, x1].astype(np.float32)
    wx = x - x0
    wy = y - y0
    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy
    val = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    return val.astype(np.uint8)

# main function
# notice you cannot run this main function until you implement the above transform() function 
if __name__ == '__main__':

    # load the image in data
    filename = 'data/000006-color.jpg'
    im = cv2.imread(filename)
    
    # image height and width
    height = im.shape[0]
    width = im.shape[1]
    
    # 2D translation
    T1 = np.eye(3, dtype=np.float32)
    T1[0, 2] = 50
    T1[1, 2] = 100
    im_1 = transform(im, T1)
    print('2D translation')
    print(T1)
    
    # 2D rotation
    R = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
    T2 = np.eye(3, dtype=np.float32)
    T2[:2, :] = R
    im_2 = transform(im, T2)
    print('2D rotation')
    print(T2)
    
    # 2D rigid transformation: 2D rotation + 2D transformation
    T3 = np.matmul(T1, T2)
    im_3 = transform(im, T3)
    print('2D rigid transform')
    print(T3)
    
    # 2D affine transformation
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    M = cv2.getAffineTransform(pts1, pts2)
    T4 = np.eye(3, dtype=np.float32)
    T4[:2, :] = M
    print('Affine transform')
    print(T4)
    im_4 = transform(im, T4)
    
    # 2D perspective transformation
    pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
    pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
    T5 = cv2.getPerspectiveTransform(pts1, pts2)
    print('Perspective transform')
    print(T5)
    im_5 = transform(im, T5)
    
    # show the images
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])        
    ax.set_title('original image')
    
    ax = fig.add_subplot(2, 3, 2)    
    plt.imshow(im_1[:, :, (2, 1, 0)])        
    ax.set_title('translated image')
    
    ax = fig.add_subplot(2, 3, 3)    
    plt.imshow(im_2[:, :, (2, 1, 0)])        
    ax.set_title('rotated image')
    
    ax = fig.add_subplot(2, 3, 4)    
    plt.imshow(im_3[:, :, (2, 1, 0)])
    ax.set_title('rigid transformed image')   
    
    ax = fig.add_subplot(2, 3, 5)
    plt.imshow(im_4[:, :, (2, 1, 0)])        
    ax.set_title('affine transformed image')
    
    ax = fig.add_subplot(2, 3, 6)    
    plt.imshow(im_5[:, :, (2, 1, 0)])        
    ax.set_title('perspective transformed image') 
    
    plt.show()
