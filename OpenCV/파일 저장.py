import cv2
img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)# 흑백 이미지
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('img_save.jpg',img)
print(result)

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)# 흑백 이미지
result = cv2.imwrite('img_save.jpg',img)# 저장

# 동영상 저장
import cv2
cap = cv2.VideoCapture('video.mp4')

# 코덱 정의
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# 프레임 크기, FPS
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output.avi',fourcc,fps,(width,height))
# 저장 파일명, 코덱, fps, 크기

while cap.isOpened():
    ret, frame = cap.read()

    if not ret :
        break

    out.write(frame)# 영상 데이터만 저장
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
