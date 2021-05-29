import cv2
import Preprocessing_img as pre
import numpy as np

img_org = cv2.imread('./forV4/test/2.jpg')
filter1 = pre.text_filter(img_org)
result = cv2.multiply(filter1, 1.5)

cv2.imshow('result', pre.resize_perten(filter1, 2))

# cv2.imwrite('result.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
