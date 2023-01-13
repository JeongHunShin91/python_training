# 가우시안 블러

import cv2
img = cv2.imread('img.jpg')

kernel_3 = cv2.GaussianBlur(img, (3,3),0)
kernel_5 = cv2.GaussianBlur(img, (5,5),0)
kernel_7 = cv2.GaussianBlur(img, (7,7),0)

cv2.imshow('3',kernel_3)
cv2.imshow('5',kernel_5)
cv2.imshow('7',kernel_7)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 표준편차에 따른 흐림
import cv2
img = cv2.imread('img.jpg')

sigma_3 = cv2.GaussianBlur(img, (0,0),1) # sigmax = 가우시안 커널의 x 방향의 표준편차
sigma_5 = cv2.GaussianBlur(img, (0,0),2)
sigma_7 = cv2.GaussianBlur(img, (0,0),3)

cv2.imshow('3',sigma_3)
cv2.imshow('5',sigma_5)
cv2.imshow('7',sigma_7)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()