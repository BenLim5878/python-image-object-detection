import cv2
import numpy as np

def dilate(img):
    img_dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
    # img_blurred = cv2.medianBlur(img_dilated, 15)
    # diff_img = 255 - cv2.absdiff(img, img_dilated)
    # norm_img = diff_img.copy()
    # cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_dilated

def addborder(img,border_size):
    img_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=(255,255,255))
    return img_border

def gaussianblur(img, kernel_size):
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img_blur

def gray(img):
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def removebackground(img):
    img_dilated = dilate(img)
    img_blur = cv2.medianBlur(img_dilated, 21)
    img_diff = 255 - cv2.absdiff(img, img_blur)
    return img

def scale(img, target_size):
    image_width = img.shape[1]
    image_height = img.shape[0]

    ratio = target_size / image_width

    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    res = np.zeros_like(img)
    res[:, :, 2] = 255
    return img;

def denoise(img, denoise_val):
    img_denoised = cv2.fastNlMeansDenoising(img, None, denoise_val, 7, 21)
    return img_denoised

def histogramEQ(img):
    img_eq = cv2.equalizeHist(img)
    return img_eq

def sharpen(img):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel_sharpening)
    return img_sharpen