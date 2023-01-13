import cv2
img = cv2.imread('img.jpg')# 해당 경로의 파일 읽어오기
cv2.imshow('img', img) # ing 라는 이름의 창에 img 를 표시
cv2.waitKey(0) # 지정된 시간 동안 사용자 키 입력 대기
cv2.destroyAllWindows() # 모든창 닫기

# 읽기 옵션
# cv2.IMREAD_COLOR : 컬러 이미지, 투명영역은 무시(기본값)
# cv2.IMREAD_GRAYSCALE : 흑백 이미지
# cv2.IMREAD_UNCHANGED : 투명영역까지 포함

img_color = cv2.imread('img.jpg',cv2.IMREAD_COLOR)
img_gray = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('img.jpg',cv2.IMREAD_UNCHANGED)

cv2.imshow('img_color',img_color)
cv2.imshow('img_gray',img_gray)
cv2.imshow('img_unchanged',img_unchanged)

cv2.waitKey(0) # 지정된 시간 동안 사용자 키 입력 대기
cv2.destroyAllWindows() # 모든창 닫기

## shape
# 이미지의 높이, 넓이, 채널
img = cv2.imread('img.jpg')
img.shape