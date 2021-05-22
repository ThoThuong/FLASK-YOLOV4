from __future__ import print_function
import cv2
import numpy as np


def resize_perten(img, per_ten):
    scale_percent = 10 * per_ten  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def resize(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def resize_with_max(img, max_size=1500):
    max_size_of_img = max(img.shape[1], img.shape[0])
    width = int(img.shape[1] * (max_size / max_size_of_img))
    height = int(img.shape[0] * (max_size / max_size_of_img))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


# def img_to_binary(img, max_size=1500):
#     # img_org = cv2.imread('D:/hk8_kltn/detect_how_similar_images_are/images/s11.jpg')
#     print('original shape', img.shape)
#     img_org = resize_with_max(img, max_size)
#     print('after resized', img_org.shape)
#     img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
#     img_denoise = cv2.fastNlMeansDenoising(img_gray, None, 20, 7, 21)
#     # img_blur = cv2.medianBlur(img_gray, 5)
#     # img_blur = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)

#     return cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
def img_to_binary(img, max_size=1500):
    # img_org = cv2.imread('D:/hk8_kltn/detect_how_similar_images_are/images/s11.jpg')
    # print('original shape', img.shape)
    img_org = resize_with_max(img, max_size)
    # print('after resized', img_org.shape)
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_denoise = cv2.fastNlMeansDenoising(img_gray, None, 20, 7, 21)
    # img_blur = cv2.medianBlur(img_gray, 5)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)

    # return cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpen = cv2.filter2D(img_denoise, -1, sharpen_kernel)
    ret, bw_img = cv2.threshold(
        img_sharpen, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw_img


def gamma_correction(img, gamma=0.7):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)
