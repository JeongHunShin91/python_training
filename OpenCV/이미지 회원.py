# 이미지 회전
import cv2
img = cv2.imread('img_save.jpg')

# 90도 회줜
rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('img',img)
cv2.imshow('rotate_90',rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 90도 회줜
rotate_180 = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow('img',img)
cv2.imshow('rotate_180',rotate_180)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 시계 반대 방향도 회줜
rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('img',img)
cv2.imshow('rotate_270',rotate_270)
cv2.waitKey(0)
cv2.destroyAllWindows()