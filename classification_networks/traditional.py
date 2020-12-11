import cv2
import numpy as np
import os



def mask(olaf):
    gray = cv2.cvtColor(olaf, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    thresh1 = cv2.Canny(blur,70,120)
    thresh1 = cv2.GaussianBlur(thresh1,(5,5),1)

    cnt = sorted(cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

    mask = np.zeros((olaf.shape[0], olaf.shape[1]), dtype=np.uint8)

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dst = cv2.bitwise_and(olaf,olaf, mask=mask)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    dst_gray[ dst_gray > 140 ] = 0
    dst_erode = cv2.erode(dst_gray,(3, 3), iterations=2)

    cv2.imshow('edge', dst_erode)
    cv2.waitKey()


