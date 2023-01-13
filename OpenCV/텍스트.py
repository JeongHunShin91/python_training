import cv2
import numpy as np

img = np.zeros((480,640,3), dtype = np.uint8)

scale = 2
color = (255,255,255)
thickness = 1

#보통의 크기의 산 세리프 글꼴
cv2.putText(img,"Nado Simplex",(20, 50), cv2.FONT_HERSHEY_SIMPLEX,scale, color, thickness)
# 그릴 위치, 텍스트 내용, 시작위치, 폰트종류, 크기, 색깔, 두께
# 작은 크기의 산세리프 글꼴
cv2.putText(img,"Nado Simplex",(20, 150), cv2.FONT_HERSHEY_PLAIN,scale, color, thickness)
# 필기체 스타일 글꼴
cv2.putText(img,"Nado Simplex",(20, 250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,scale, color, thickness)
# 보통 크기의 산 세리프 글꼴
cv2.putText(img,"Nado Simplex",(20, 350), cv2.FONT_HERSHEY_TRIPLEX,scale, color, thickness)
# 기울기
cv2.putText(img,"Nado Simplex",(20, 450), cv2.FONT_ITALIC,scale, color, thickness)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 한글위회방법
from PIL import ImageFont, ImageDraw, Image

def mytext(src, text,pos,font_size,font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font,fill = font_color)
    return np.array(img_pil)

img = np.zeros((480,640,3), dtype = np.uint8)

FONT_SIZE = 30
color = (255,255,255)

#보통의 크기의 산 세리프 글꼴
img = mytext(img, "지영아 힘내 넌 할 수 있어", (20, 50), FONT_SIZE, color)
# 그릴 위치, 텍스트 내용, 시작위치, 폰트종류, 크기, 색깔, 두께
# 작은 크기의 산세리프 글꼴

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()