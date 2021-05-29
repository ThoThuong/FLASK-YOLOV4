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


def text_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 100, 200)
    inv_gray = cv2.bitwise_not(img_gray)
    sub_img = cv2.bitwise_not(cv2.multiply(cv2.subtract(img_gray, canny), 2.5))
    addition_img = cv2.bitwise_not(cv2.add(sub_img, inv_gray))
    contras_img = cv2.multiply(addition_img, 1.3)
    result = cv2.subtract(contras_img, canny)
    return result

def text_filter_canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img_gray, 100, 200)
    inv_canny = cv2.bitwise_not(canny)

    return inv_canny

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
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
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
