import cv2
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('img_save.jpg')

hist = cv2.calcHist([src], [0], None, [
    256], [0, 256])
plt.subplot(1, 2, 1)
plt.title('hist')
plt.plot(hist)

dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)

hist2 = cv2.calcHist([dst], [0], None, [
    256], [0, 256])
plt.subplot(1, 2, 2)
plt.title('hist2')
plt.plot(hist2)

cv2.imshow('dst', dst)
cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()