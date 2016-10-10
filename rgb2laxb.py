import cv2
import numpy as np

img = cv2.imread('watch2.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
cv2.imshow('image',img2)
cv2.waitKey()
cv2.destroyAllWindows()


