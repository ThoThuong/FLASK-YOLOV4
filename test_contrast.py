import cv2
import Preprocessing_img as pre
import numpy as np

img_org = cv2.imread('./forV4/test/3.jpg')
thresh_img = pre.cal_thresh(img_org)
low_brightness_img = pre.cal_low_brightness(img_org, thresh_img)
result = cv2.subtract(low_brightness_img, thresh_img)
result = pre.thresholding(result)

cv2.imshow('img_org', pre.resize_perten(img_org, 2))
cv2.imshow('thresh_img', pre.resize_perten(thresh_img, 2))
cv2.imshow('low_brightness_img', pre.resize_perten(low_brightness_img, 2))
# cv2.imshow('result', pre.resize_perten(result, 5))
cv2.imshow('result', pre.resize_perten(pre.text_filter(img_org), 5))

# cv2.imwrite('result.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
