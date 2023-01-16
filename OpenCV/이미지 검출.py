import cv2
img = cv2.imread('snowman.png')

def empty(pos) :
    pass

name = 'Trackbar'
cv2.namedWindow(name)

canny = cv2.Canny(img,150,200)

cv2.imshow('canny',canny)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()