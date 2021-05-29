import cv2
import Preprocessing_img as pre
import numpy as np

img_org = cv2.imread('./forV4/test/2.jpg')
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
canny = pre.cal_canny(img_gray)
negative_canny = cv2.bitwise_not(canny)
sub_img1 = cv2.bitwise_not(img_gray)
sub_img2 = cv2.multiply(cv2.subtract(img_gray, canny), 0.5)
sub_img3 = cv2.bitwise_not(cv2.multiply(cv2.subtract(img_gray, canny), 2.5))
add_img = cv2.add(sub_img2, 0)
thresh_img = pre.thresholding(add_img)
add_img_cann1 = cv2.subtract(negative_canny, thresh_img)
dilate = pre.cal_dilate(thresh_img,7)


dilate = pre.cal_erode(dilate,20)



negative_sub2 = cv2.multiply(cv2.subtract(img_gray, canny), 2)

add_img_cann2 = cv2.bitwise_not(cv2.subtract(negative_sub2, dilate))

result = cv2.subtract(add_img_cann2, thresh_img)
result = pre.thresholding(result)
# dilate=pre.dilate(dilate)
# dilate=pre.dilate(dilate)
# dilate=pre.erode(dilate)
# dilate=pre.erode(dilate)
# dilate=pre.erode(dilate)
# dilate=pre.erode(dilate)

# add_img_cann2 = cv2.add(add_img_cann1,canny)


cv2.imshow('img_org', pre.resize_perten(img_org, 2))
cv2.imshow('img_gray', pre.resize_perten(img_gray, 2))
cv2.imshow('img_cann', pre.resize_perten(canny, 2))
cv2.imshow('n_canny', pre.resize_perten(negative_canny, 2))
cv2.imshow('img_sub2', pre.resize_perten(sub_img2, 2))
cv2.imshow('img_sub1', pre.resize_perten(sub_img1, 2))
cv2.imshow('add_img', pre.resize_perten(add_img, 2))
cv2.imshow('thresh_img', pre.resize_perten(thresh_img, 2))
cv2.imshow('add_img_cann1', pre.resize_perten(add_img_cann1, 2))
cv2.imshow('add_img_cann2', pre.resize_perten(add_img_cann2, 2))
cv2.imshow('dilate', pre.resize_perten(dilate, 2))
cv2.imshow('negative_sub2', pre.resize_perten(negative_sub2, 2))
cv2.imshow('result', pre.resize_perten(result, 2))
# cv2.imshow('add_img_cann2', pre.resize_perten(add_img_cann2, 2))

cv2.imwrite('result.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
