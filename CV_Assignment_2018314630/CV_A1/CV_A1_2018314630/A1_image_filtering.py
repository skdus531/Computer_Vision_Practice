import numpy as np
from cv2 import cv2
import os
import time

def get_gaussian_filter_1d(size, sigma):
    alpha = size//2
    gaussian_filter = []
    for i in range(-alpha, alpha+1):
        x1 = 2*np.pi*(sigma**2)
        x2 = np.exp(-(i**2)/(2* (sigma**2)))
        gaussian_filter.append((1/x1)*x2)
    gaussian_filter /= np.sum(gaussian_filter)
    return np.reshape(gaussian_filter, (1,gaussian_filter.shape[0])), np.reshape(gaussian_filter,(-1,1))


def get_gaussian_filter_2d(size, sigma):
    guassian_filter = np.outer(get_gaussian_filter_1d(size,sigma)[0],
        get_gaussian_filter_1d(size,sigma)[0])
    guassian_filter /= np.sum(guassian_filter)
    return np.array(guassian_filter)
    

def cross_correlation_1d(img, kernel):
    size = np.shape(kernel)
    
    if size[0] == 1 :  #vertical kernel 
        alpha = size[1]//2
        padding_img = boundaries(img, (0,alpha))
        stride_img = strides(padding_img,(1,size[1]))
        filtered_img = einsum(stride_img,kernel)
        return filtered_img

    else:   #horizontal kernel
        alpha = size[0]//2
        padding_img = boundaries(img, (alpha,0))
        stride_img = strides(padding_img,(size[0],1))
        filtered_img = einsum(stride_img,kernel)
        return filtered_img
        

def cross_correlation_2d(img, kernel):
    size = np.shape(kernel)
    alpha = size[0]//2
    padding_img = boundaries(img, (alpha, alpha))
    stride_img = strides(padding_img, kernel.shape)
    filtered_img = einsum(stride_img,kernel)
    return filtered_img

def boundaries(img, alpha):
    row = img.shape[0] + alpha[0]
    col = img.shape[1] + alpha[1]

    res = np.empty((row+alpha[0], col+alpha[1]))
    
    res[alpha[0]:row, alpha[1]:col] = img
    res[0:alpha[0], 0:alpha[1]] = img[0, 0]  # left up
    res[0:alpha[0], col:] = img[0, img.shape[1]-1] # right up
    res[row:, 0:alpha[1]] = img[img.shape[0]-1, 0] # left bottom
    res[row:, col:] = img[img.shape[0]-1, img.shape[1]-1] # right bottom

    res[0:alpha[0], alpha[1]:col] = img[0, :] # up
    res[row:, alpha[1]:col] = img[img.shape[0]-1,:] # bottom
    res[alpha[0]:row, 0:alpha[1]] = np.reshape(img[:, 0], (-1, 1)) # left
    res[alpha[0]:row, col:] = np.reshape(img[:, img.shape[1]-1], (-1, 1)) # right

    return res

def strides(img,size):
    return np.lib.stride_tricks.as_strided(img,
        shape=(img.shape[0] - size[0] + 1, img.shape[1] - size[1] + 1, size[0], size[1],),
        strides=(img.strides[0], img.strides[1], img.strides[0], img.strides[1],),
        writeable=False,)


def einsum(img,kernel):
    return np.einsum("ijkl,kl->ij", img, kernel)

if __name__ == "__main__":

    # 1-2. c) 
    print("1D Gaussian Kernel (5,1)")
    print(get_gaussian_filter_1d(5,1)[0],"\n")
    print("2D Gaussian Kernel (5,1)")
    print(get_gaussian_filter_2d(5,1),"\n")  

    # 1-2. d)
    INPUT_IMAGE_FILE_NAME = ["shapes.png", "lenna.png"]
    result = []
    for image in INPUT_IMAGE_FILE_NAME:
        print("=========================INPUT IMAGE FILE :",image,"=========================\n")
        img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
                
        for size in [5, 11, 17]:
            same_size = []
            for sigma in [1, 6, 11]:
                kernel = get_gaussian_filter_2d(size,sigma)
                filtered_img = cross_correlation_2d(img,kernel)
                
                caption = str(size)+"x"+str(size)+" s="+str(sigma)
                cv2.putText(filtered_img, caption, (15,25), cv2.FONT_HERSHEY_DUPLEX,1, 0)

                if sigma == 1: same_size = filtered_img
                else : same_size = np.concatenate((same_size,filtered_img), axis=1)

            if size == 5:result = same_size
            else: result = np.concatenate((result,same_size), axis=0)
        
        cv2.imshow("Gaussian Filtering Image_"+image,result/255)
        cv2.waitKey(0)

        # 1-2 e)
        start_time = time.time()
        G_1D = cross_correlation_1d(img,get_gaussian_filter_1d(5,1)[1])
        G_1D = cross_correlation_1d(G_1D,get_gaussian_filter_1d(5,1)[0])
        G_2D = cross_correlation_2d(img,get_gaussian_filter_2d(5,1))
        print("Computational Times of 1D and 2D Filterings : ", time.time()-start_time,"\n")
        print("Sum of Intensity Difference : ", np.sum(np.abs(G_2D-G_1D)),"\n\n")

        cv2.imshow("Difference Map_"+image,np.subtract(G_2D,G_1D))
        cv2.waitKey(0)
        # 1-2 f)
        if not os.path.exists("./result"):
            os.makedirs("./result")
        cv2.imwrite("./result/part_1_gaussian_filtered_"+image, result)
    cv2.destroyAllWindows()