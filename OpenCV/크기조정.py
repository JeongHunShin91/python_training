import cv2

# 고정크기로설정
import cv2
img = cv2.imread('img.jpg')
dst = cv2.resize(img,(400,500))

cv2.imshow('img',img)
cv2.imshow('resize',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 비율로 설정
img = cv2.imread('img.jpg')
dst = cv2.resize(img,None, fx=0.5,fy =0.5)
# x,y 비율 정의( 0.5배로 축소)

cv2.imshow('img',img)
cv2.imshow('resize',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 보간법 적용 축소
img = cv2.imread('img.jpg')
dst = cv2.resize(img,None, fx=0.5,fy =0.5,interpolation = cv2.INTER_AREA)
# x,y 비율 정의( 0.5배로 축소)

cv2.imshow('img',img)
cv2.imshow('resize',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 보간법 적용 확대
img = cv2.imread('img_save.jpg')
dst = cv2.resize(img,None, fx=0.5,fy =0.5,interpolation = cv2.INTER_CUBIC)
# x,y 비율 정의( 0.5배로 축소)

cv2.imshow('img',img)
cv2.imshow('resize',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 동영상크기 축소
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret :
        break

    frame_resized = cv2.resize(frame, None, fx = 0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('video',frame_resized)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 동영상 크기 픽스에 맞춰서 축소
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret :
        break

    frame_resized = cv2.resize(frame,(400,500))
    cv2.imshow('video',frame_resized)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()