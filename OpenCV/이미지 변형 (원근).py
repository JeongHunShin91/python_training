import cv2
import numpy as np

# 사다리꼴 이미지 펼치기
img = cv2.imread('newspaper.jpg')

width, height = 640, 240 # 가로크기 640, 세로크기 240 결과물 출력

src = np.array([[511,352],[1008,345],[1122,584],[455,594]], dtype = np.float32) # input 4개지정
dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = np.float32)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix
result = cv2.warpPerspective(img, matrix,(width, height)) # matrix 대로변환

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 회전된 이미지 똑바로 만들기
img = cv2.imread('porker.jpg')

width, height = 530, 710 # 가로크기 530, 세로크기 710 결과물 출력

src = np.array([[702,143],[1133,414],[726,1007],[276,700]], dtype = np.float32) # input 4개지정
dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = np.float32)

matrix = cv2.getPerspectiveTransform(src, dst) # Matrix
result = cv2.warpPerspective(img, matrix,(width, height)) # matrix 대로변환

cv2.imshow('img',img)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 미니 프로젝트 : 반자동 문서 스캐너
# 마우스 이벤트 등록
import cv2
def mouse_handler(event, x, y, flags, paran):
    if event == cv2.EVENT_LBUTTONDOWN : # 마우스 왼쪽 버튼 DOWN
        print('왼쪽버튼 down')
        print(x,y)
    elif event == cv2.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 UP
        print('왼쪽버트 up')
        print(x,y)
    elif event == cv2.EVENT_LBUTTONDBLCLK: # 마우스 왼쪽 버튼 더블 클릭
        print('왼쪽 버튼 더블')
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     print('마우스이동')
    elif event == cv2.EVENT_RBUTTONDOWN:
        print('오른쪽 버튼 Down')

img = cv2.imread('porker.jpg')
cv2.namedWindow('img') #img 란 이름의 원도우를 먼저 만들어두는 것 여기에 마우스 이벤트를 처리하기 위한 핸들러 적용
cv2.setMouseCallback('img', mouse_handler)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 프로젝트
import cv2
import numpy as np

point_list = []
src_img = cv2.imread('porker.jpg')

color = (255,0,255)
THICKNESS = 3
drawing =False # 선을 그릴지 여주

def mouse_handler(event, x, y, flags, paran):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN : # 마우스 왼쪽 버튼 DOWN
        drawing = True # 선을 그리기 시작
        point_list.append((x,y))

    # if drawing :
    #     prev_point = None # 직선의 시작점
    for point in point_list:
        cv2.circle(src_img, point, 15, color, cv2.FILLED)
            # if prev_point :
            #     cv2.line(src_img,prev_point, color, THICKNESS, cv2.LINE_AA)
            # prev_point = point

    if len(point_list) == 4:
        show_result()

    cv2.imshow('img',src_img)

def show_result():
    width, height = 530, 710  # 가로크기 530, 세로크기 710 결과물 출력

    src = np.float32(point_list)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)  # Matrix
    result = cv2.warpPerspective(src_img, matrix, (width, height))  # matrix 대로변환

    cv2.imshow('result',result)

cv2.namedWindow('img') #img 란 이름의 원도우를 먼저 만들어두는 것 여기에 마우스 이벤트를 처리하기 위한 핸들러 적용
cv2.setMouseCallback('img', mouse_handler)
cv2.imshow('img',src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()