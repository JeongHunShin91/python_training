import cv2
# 흑백 읽어오기
img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindws()

# 흑백으로 저장
img = cv2.imread('img.jpg')

dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('dst',dst)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindws()
