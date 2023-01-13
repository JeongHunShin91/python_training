import cv2
img = cv2.imread('img_save.jpg')

# 좌우 대칭
flip_horizontal = cv2.flip(img, 1) # flipcode >0 : 좌주 대칭

cv2.imshow('img',img)
cv2.imshow('flip',flip_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 상하 대칭
flip_vertical = cv2.flip(img,0) # flipcode ==0

cv2.imshow('img',img)
cv2.imshow('flip',flip_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 상하좌우 대칭
flip_both = cv2.flip(img,-1) # flipcode ==0

cv2.imshow('img',img)
cv2.imshow('flip',flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()