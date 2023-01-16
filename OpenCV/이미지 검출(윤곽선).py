import cv2
img = cv2.imread('card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)# 윤곽선 검출

COLOR = (0,200,0)
cv2.drawContours(target_img, contours, -1,COLOR, 2)

cv2.imshow('img',img)
cv2.imshow('gray',gray)
cv2.imshow('otsu',otsu)
cv2.imshow('contour',target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 윤석선 찾기 모드
import cv2
img = cv2.imread('card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# 윤곽선 검출

COLOR = (0,200,0)
cv2.drawContours(target_img, contours, -1,COLOR, 2)

cv2.imshow('img',img)
cv2.imshow('gray',gray)
cv2.imshow('otsu',otsu)
cv2.imshow('contour',target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 경계 사각형
import cv2
img = cv2.imread('card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# 윤곽선 검출

COLOR = (0,200,0)
for cnt in contours :
    x, y, width, height = cv2.boundingRect(cnt)
    cv2.rectangle(target_img, (x,y), (x + width, y+height), COLOR,2)

cv2.imshow('img',img)
cv2.imshow('gray',gray)
cv2.imshow('otsu',otsu)
cv2.imshow('contour',target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 면적
import cv2
img = cv2.imread('card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# 윤곽선 검출

COLOR = (0,200,0)
for cnt in contours :
    if cv2.contourArea(cnt) > 25000:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2)

cv2.imshow('img',img)
cv2.imshow('contour',target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 개별 카드 추출하여 파일저장
import cv2
img = cv2.imread('card.png')
target_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# 윤곽선 검출

COLOR = (0,200,0)

idx=1
for cnt in contours :
    if cv2.contourArea(cnt) > 25000:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(target_img, (x, y), (x + width, y + height), COLOR, 2)

        crop = img[y:y+height, x:x+width]
        cv2.imshow(f'card_crop_{idx}',crop)
        cv2.imwrite(f'card_crop_{idx}.png',crop)
        idx +=1

cv2.imshow('img',img)
cv2.imshow('contour',target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()