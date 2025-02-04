import cv2
import numpy as np

img = cv2.imread('1474532b93f44948152e7b798ba876935081df4a_2_1035x318.jpeg')

kernel = np.ones((6,6),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)

gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
#
ret,gray = cv2.threshold(gray,160,255,0)
gray2 = gray.copy()
mask = np.zeros(gray.shape,np.uint8)

contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 400:
        approx = cv2.approxPolyDP(cnt,
                                  0.005 * cv2.arcLength(cnt, True), True)
        if(len(approx) >= 5):
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
res = cv2.bitwise_and(gray2,gray2,mask)
cv2.imwrite('output.png',img)