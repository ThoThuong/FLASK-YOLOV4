import cv2
import Preprocessing_img as pre
import numpy as np

img_org = cv2.imread('./forV4/test/4.jpg')
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
canny = pre.canny(img_gray)
negative_canny = cv2.bitwise_not(canny)
sub_img1 = cv2.bitwise_not(img_gray)
sub_img2 = cv2.bitwise_not(cv2.multiply(cv2.subtract(img_gray, canny), 2.5))
sub_img = cv2.bitwise_not(cv2.add(sub_img2, sub_img1))
remove_dark = cv2.bitwise_not(cv2.subtract(sub_img, img_gray))
result = cv2.subtract(cv2.multiply(remove_dark, 1), canny)

img_b = pre.img_to_binary(img_org)
# th3 = cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('img_org', pre.resize_perten(img_org, 2))
cv2.imshow('img_gray', pre.resize_perten(img_gray, 2))
cv2.imshow('img_cann', pre.resize_perten(canny, 2))
cv2.imshow('n_canny', pre.resize_perten(negative_canny, 2))
cv2.imshow('result', pre.resize_perten(result, 2))
cv2.imshow('img_sub2', pre.resize_perten(sub_img2, 2))
cv2.imshow('img_sub1', pre.resize_perten(sub_img1, 2))
cv2.imshow('img_sub', pre.resize_perten(sub_img, 2))
cv2.imshow('remove_dark', pre.resize_perten(remove_dark, 2))
cv2.imshow('thresh', pre.resize_perten(pre.thresholding(sub_img), 2))
cv2.imshow('img_b', pre.resize_perten(img_b, 2))


cv2.waitKey(0)
cv2.destroyAllWindows()
