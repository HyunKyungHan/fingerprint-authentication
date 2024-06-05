################## PREPROCESSING UTILS ###################
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as skelt

def skeletonize(image_input):
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)
    skeleton = skelt(image)
    output[skeleton] = 255
    cv2.bitwise_not(output, output)
    return output

def enhance_skelt(img, ks=3):
    ks = 3
    # kernel = np.ones((ks, ks)) / (ks*ks)
    kernel = np.array((
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ))
    
    rows, cols = img.shape[:2]
    img_filtered = np.zeros(img.shape, dtype=np.uint8)
    
    for i in range(1,rows-1):
            for j in range(1,cols-1):
                result = 0
                for k_i in range(0, len(kernel)):
                    for k_j in range(0, len(kernel[0])):
                        result += img[i-ks//2+k_i, j-ks//2+k_j] * kernel[(len(kernel)-1)-k_i, (len(kernel)-1)-k_j]  # convolution
                img_filtered[i, j] = int(result) if result > 0 else 0
    
    return img_filtered

def read_and_preprocess(img, idx, save_image):
    # 1. READ IMAGE
    ks = 3
    # kernel = np.ones((ks, ks)) / (ks*ks)
    kernel = np.array((
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ))

    rows, cols = img.shape[:2]
    img_filtered = np.zeros(img.shape, dtype=np.uint8)

    for i in range(1,rows-1):
            for j in range(1,cols-1):
                # block = img[i-1:i+2,j-1:j+2]
                result = 0
                for k_i in range(0, len(kernel)):
                    for k_j in range(0, len(kernel[0])):
                        # result += img1[i, j] * kernel[k_i, k_j]  # correlation
                        result += img[i-ks//2+k_i, j-ks//2+k_j] * kernel[(len(kernel)-1)-k_i, (len(kernel)-1)-k_j]  # convolution
                img_filtered[i, j] = int(result) if result > 0 else 0


    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(img_filtered, kernel2)

    skel = skeletonize(dst)
    skel = 255 - skel

    if save_image:
        merged = np.hstack((img_filtered, dst, skel))
        plt.figure(figsize=(15, 17), facecolor='white')
        plt.title('Binarization & Skeletonization Result')
        plt.imsave(f'results/preprocessed/{idx}_preprocessed.png', merged, cmap='gray')

    return skel