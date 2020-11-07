import numpy as np
from cv2 import cv2
import time
import os
import A1_image_filtering as filtering

THRESHOLD = 0.1

def compute_corner_response(img):
    filtered_img = filtering.cross_correlation_2d(img, 
        filtering.get_gaussian_filter_2d(7,1.5))
    
    sobel_xx = [[-1, 0, 1]]
    sobel_xy = [[1], [2], [1]]
    sobel_yx = [[1, 2, 1]]
    sobel_yy = [[-1], [0], [1]]

    dx = filtering.cross_correlation_1d(filtered_img,sobel_xx)
    dx = filtering.cross_correlation_1d(dx,sobel_xy)
    dy = filtering.cross_correlation_1d(filtered_img,sobel_yx)
    dy = filtering.cross_correlation_1d(dy,sobel_yy)

    #sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    #sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    #dx = filtering.cross_correlation_2d(filtered_img,sobel_x)
    #dy = filtering.cross_correlation_2d(filtered_img,sobel_y)

    Ix2 = dx**2
    IxIy = dx * dy
    Iy2 = dy**2

    window = np.array([1]*25).reshape((5,5))
    alpha = window.shape[0]//2
    Ix2 = filtering.einsum(filtering.strides(boundaries(Ix2, alpha),window.shape),window)
    IxIy = filtering.einsum(filtering.strides(boundaries(IxIy, alpha),window.shape),window)
    Iy2 = filtering.einsum(filtering.strides(boundaries(Iy2, alpha),window.shape),window)

    M = np.empty((Ix2.shape[0],Ix2.shape[1],2,2))
    M[:,:,0,0] = Ix2
    M[:,:,0,1] = M[:,:,1,0] = IxIy
    M[:,:,1,1] = Iy2
    
    R = np.linalg.det(M[:,:]) - 0.04*(M[:,:,0,0]+M[:,:,1,1])**2
    R[R<0] = 0
    R /= np.max(R)
    return R

def non_maximum_suppression_win(R, winSize):
    alpha = winSize//2
    filtered_R = filtering.strides(boundaries(R,alpha),(winSize,winSize))
    max_R = np.max(filtered_R,axis=(2,3))
    res = R.copy()
    res[res<THRESHOLD] = res[max_R>R] = 0
    return res


def boundaries(img, alpha):
    res = np.zeros((2*alpha+img.shape[0], 2*alpha+img.shape[1]))
    res[alpha:alpha+img.shape[0], alpha:alpha+img.shape[1]] = img
    return res

if __name__ == "__main__":
    
    INPUT_IMAGE_FILE_NAME = ["shapes.png", "lenna.png"]

    for image in INPUT_IMAGE_FILE_NAME:
        print("\n=========================INPUT IMAGE FILE :",image,"=========================\n")
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # 3-2 e)
        start_time = time.time()
        R = compute_corner_response(img)
        print("Computational Times of Corner Response :", time.time()-start_time,"\n")
        
        cv2.imshow("RAW_"+image, R)
        cv2.waitKey(0)
        if not os.path.exists("./result"):
            os.makedirs("./result")
        cv2.imwrite("./result/part_3_corner_raw_"+image,R*255)

        # 3-3 b)
        img1 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img1[R>THRESHOLD] = [0,255,0]
        cv2.imshow("THRESHOLDING_"+image, img1)
        cv2.waitKey(0)
        cv2.imwrite("./result/part_3_corner_bin_"+image, img1)

        # 3-3 d)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)    
        start_time = time.time()
        suppressed_R = non_maximum_suppression_win(R,11)
        print("Computational Times of NMS : ",time.time()-start_time,"\n")

        points = np.where(suppressed_R>THRESHOLD)
        for i in range(len(points[1])):
            img = cv2.circle(img,(points[1][i], points[0][i]),5, (0,255,0), thickness=2)

        cv2.imshow("NMS_"+image, img)
        cv2.waitKey(0)
        cv2.imwrite("./result/part_3_corner_sup_"+image,img)
    
    cv2.destroyAllWindows()