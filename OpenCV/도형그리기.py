import cv2
import numpy as np

# 세로 480 x 가로 640, 3 channel
img = np.zeros((480,640,3), dtype = np.uint8) # 스케치북 만들기
img[:] = (255,0,255) # 색칠하기
img
cv2.imshow('img', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# 일부색 칠하기
img = np.zeros((480,640,3), dtype = np.uint8)
img[100:110,200:300] = (255,255,255)# [세로영억, 가로영역]

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv.LINE_4 : 상하좌우 4 방향으로 연결된 선
# cv.LINE_8 : 대각선을 포함한 8방향으로 연결됭 선(기본값)
# cv.LINE_AA : 부드러운선

img = np.zeros((480,640,3), dtype = np.uint8) # 스케치북 만들기

color = (0,255,255) # bgr : 색깔
thickess = 3 # 두께
cv2.line(img, (50,100),(400,50),color,thickess,cv2.LINE_8)
# 그릴 위치, 시작점, 끝점, 색깔, 두꼐, 선종류
cv2.line(img, (50,200),(400,150),color,thickess,cv2.LINE_4)
cv2.line(img, (50,300),(400,250),color,thickess,cv2.LINE_AA)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 원
img = np.zeros((480,640,3), dtype = np.uint8) # 스케치북 만들기

color = (255,255,0) # bgr : 색깔
radius = 50 # 반지름
thickess = 3 # 두께

cv2.circle(img, (200, 200), radius,color,thickess,cv2.LINE_AA) # 속이빈원
# 그릴 위치, 원의 중심정, 반지름, 색깔, 두꼐, 선의 종류
cv2.circle(img, (400, 400), radius,color, cv2.FILLED,cv2.LINE_AA) # 속이찬원

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 사각형
img = np.zeros((480,640,3), dtype = np.uint8) # 스케치북 만들기

color = (0,255,0) # bgr : 색깔
thickess = 3 # 두께

cv2.rectangle(img, (100, 100), (200,200),color, thickess)# 속이 빈 사각형
# 그릴위치, 왼쪽위 좌표, 오른쪽아래 좌표, 색깔, 두께
cv2.rectangle(img, (300, 100), (400,300),color, cv2.FILLED)# 꽉찬 사각형

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 다각형
img = np.zeros((480,640,3), dtype = np.uint8) # 스케치북 만들기

color = (0,0,255) # bgr : 색깔
thickess = 3 # 두께

pts1 = np.array([[100,100],[200,100],[100,200]])
pts2 = np.array([[200,100],[300,300],[300,200]])

# cv2.polylines(img,[pts1],True, color,thickess, cv2.LINE_AA)
# cv2.polylines(img,[pts2],True, color,thickess, cv2.LINE_AA)
cv2.polylines(img,[pts1, pts2],True, color,thickess, cv2.LINE_AA)# 속이빈 다각형
# 그릴 위치, 그릴 좌표, 닫힘여부, 색깔, 두께, 선종류

pts3 = np.array([[[100,300,],[200,300],[100,400]],[[200,300],[300,300],[300,400]]])
cv2.fillPoly(img, pts3,color,cv2.LINE_AA) # 꽉찬 다각형
# 그릴 위치, 그릴 좌표, 색깔, 선종류

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

