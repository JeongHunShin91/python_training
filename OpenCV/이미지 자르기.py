import cv2
img = cv2.imread('img.jpg')
# img.shape

crop = img[100:200, 200:400] # 기준으로 짜름

cv2.imshow('img',img)
cv2.imshow('crop',crop)
cv2.waitKey(0)
cv2.destroyAllWindws()

# 기존 윈도우에 표시
img = cv2.imread('img.jpg')

crop = img[300:600, 500:1000] # 기준으로 짜름
img[600:900, 100:600] = crop
cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindws()