import numpy as np
from cv2 import cv2
import A1_image_filtering as filtering
import os
import time

def compute_image_gradient(img):
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
    
    # sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    # dx = filtering.cross_correlation_2d(filtered_img,sobel_x)
    # dy = filtering.cross_correlation_2d(filtered_img,sobel_y)
    
    g_mag = np.array(np.sqrt(dx**2+dy**2))
    g_dir = np.arctan2(dy,dx)

    return g_mag, g_dir
    
def non_maximum_suppression_dir(mag, dir):
    row, col = mag.shape
    res = np.zeros((row, col), dtype=np.int32)
    
    angle = (dir * 180./np.pi)/45
    angle[angle<0] += 4
    angle[angle>3.5] = 0
    angle = np.around(angle, decimals=1).astype(np.int32)

    forward_idx = np.array([[0,1], [-1, -1], [1, 0], [1, -1]])
    backward_idx = np.array([[0,-1], [1, 1], [-1, 0], [-1, 1]])

    for i in range(row-1):
        for j in range(col-1): 
            val = angle[i,j]
            forward = mag[i+forward_idx[val,0], j+forward_idx[val,1]]
            backward = mag[i+backward_idx[val,0], j+backward_idx[val,1]]

            if (mag[i,j] >= forward) and (mag[i,j] >= backward): 
                res[i,j] = mag[i,j]
            else: res[i,j] = 0

    return res

if __name__ == "__main__":
    INPUT_IMAGE_FILE_NAME = ["shapes.png", "lenna.png"]
    
    for image in INPUT_IMAGE_FILE_NAME:
        print("\n=========================INPUT IMAGE FILE :",image,"=========================\n")
        img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)

        # 2-2 d)
        start_time = time.time()
        mag, dir = compute_image_gradient(img)
        print("Computational Times of Image Gradient : ", time.time()-start_time,"\n")
        
        cv2.imshow("RAW_MAG_"+image,mag/255)
        cv2.waitKey(0)
        if not os.path.exists("./result"):
            os.makedirs("./result")
        cv2.imwrite("./result/part_2_edge_raw_"+image,mag)
        
        # 2-3 d)
        start_time = time.time()
        suppressed_mag = non_maximum_suppression_dir(mag, dir)
        print("Computational Times of Non Maximum Suppression: ", time.time() - start_time,"\n\n")
        
        cv2.imshow("NMS_MAG_"+image, suppressed_mag / 255)
        cv2.imwrite("./result/part_2_edge_sup_"+image, suppressed_mag)
        cv2.waitKey(0)
    cv2.destroyAllWindows()