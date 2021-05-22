import cv2
import Preprocessing_img as pre

# img_org = cv2.imread('test/eeeeeeeeeeeeeeeeee.jpg')
img_org = cv2.imread('./forV4/test/eeeeeeeeeeeeeeeeee.jpg')
print(img_org.shape)
img_org = pre.resize_with_max(img_org, )
print(img_org.shape)
img_gamma = pre.gamma_correction(img_org,0.7)
img_gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
img_denoise = cv2.fastNlMeansDenoising(img_gray, None, 8, 7, 21)
# img_blur = cv2.medianBlur(img_gray, 5)
# img_blur = cv2.GaussianBlur(img_gray, (3, 3), cv2.BORDER_DEFAULT)


th3 = cv2.adaptiveThreshold(img_denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('img_org', pre.resize_perten(img_org, 7))
cv2.imshow('img_grey', pre.resize_perten(img_gray, 7))
cv2.imshow('filtered', pre.resize_perten(th3, 7))

cv2.waitKey(0)
cv2.destroyAllWindows()
