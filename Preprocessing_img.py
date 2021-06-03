from __future__ import print_function
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags


def by_pass_exif_orientation(image, exif):
    if exif == 3:
        image = image.rotate(180, expand=True)
    elif exif == 6:
        image = image.rotate(270, expand=True)
    elif exif == 8:
        image = image.rotate(90, expand=True)
    elif exif == 2:
        image = ImageOps.mirror(image)
    elif exif == 4:
        image = ImageOps.mirror(image)
        image = image.rotate(180, expand=True)
    elif exif == 5:
        image = ImageOps.flip(image)
        image = image.rotate(270, expand=True)
    elif exif == 7:
        image = ImageOps.flip(image)
        image = image.rotate(90, expand=True)
    return image


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


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def cal_dilate(image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


# erosion
def cal_erode(image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def cal_canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def cal_thresh(img_org):
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    canny = cal_canny(img_gray)
    # cv2.imshow('canny', resize_perten(canny, 5))
    img_blur = cv2.medianBlur(img_gray, 5)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpen = cv2.filter2D(img_blur, -1, sharpen_kernel)
    # cv2.imshow('img_sharpen', resize_perten(img_sharpen, 2))

    img_sharpen_bounding = cal_erode(cv2.subtract(img_sharpen, cal_canny(img_gray)), 1)
    img_sharpen_bounding = cal_dilate(cv2.subtract(img_sharpen_bounding, cal_canny(img_gray)), 1)

    # cv2.imshow('img_sharpen_bounding', resize_perten(img_sharpen_bounding, 5))

    sub_img2 = cv2.multiply(cv2.subtract(img_sharpen_bounding, canny), 1.1)
    # cv2.imshow('sub_img2', resize_perten(sub_img2, 2))

    add_img = cv2.add(sub_img2, 0)
    thresh_img = thresholding(add_img)
    return thresh_img


def cal_low_brightness(img_org, thresh_img):
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    canny = cal_canny(img_gray)
    dilate = cal_dilate(thresh_img, 7)
    dilate = cal_erode(dilate, 20)
    negative_sub2 = cv2.multiply(cv2.subtract(img_gray, canny), 2)
    add_img_cann2 = cv2.bitwise_not(cv2.subtract(negative_sub2, dilate))
    return add_img_cann2


def text_filter(img):
    # img=resize_with_max(img)
    thresh_img = cal_thresh(img)
    low_brightness_img=cal_low_brightness(img,thresh_img)
    result = cv2.subtract(low_brightness_img, thresh_img)
    result = thresholding(result)
    return cv2.bitwise_not(result)

def text_filter_canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 100, 200)
    inv_canny = cv2.bitwise_not(canny)

    return inv_canny
