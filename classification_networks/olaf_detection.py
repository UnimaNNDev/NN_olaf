import cv2
import numpy as np
import os
import imutils
import statistics
from random import random
import matplotlib.pyplot as plt
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage import exposure
from skimage import filters


def validation(image):
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image






# con las coodenadas se puede definir cual olaf esta más cerca de la tarjeta y definiy cual es el middle y cual edge

def count_circle_pixels(circles):  # creo que es mejor mandar la imagen
    for name in circles:
        mean = []
        for a_tuple in circles[name]:
            image = cv2.imread(name)
            image = validation(image)
            image = imutils.resize(image, width=300)
            image = exposure.rescale_intensity(image)
            #gamma_corrected = exposure.adjust_log(image, 5)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = exposure.equalize_adapthist(image, clip_limit=0.03)
            #ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_TOZERO)
            cv2.imshow('', image)
            cv2.waitKey(0)
            #image = filters.median(image, np.ones((3, 3, 3)))
            mask = np.zeros(image.shape, np.uint8)
            center = (a_tuple[0], a_tuple[1])
            radius = a_tuple[2]
            cv2.circle(mask, center, radius, 255, -1)
            # this will give you the coordinates of points inside the circle
            where = np.where(mask == 255)
            intensity_values_from_original = image[where[0], where[1]]*100
            mean_circle = statistics.median_low(intensity_values_from_original.flatten())
            mean.append(mean_circle)
            plt.hist(intensity_values_from_original)
            #plt.show()
    return mean


folder = './assets'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = './circles'
os.makedirs(folder_perspectives) if not os.path.isdir(folder_perspectives) else None

for path, filename in zip(paths, filenames):  # inte en el primer elemento de path y en el primer elemento de filename

    img = cv2.imread(path)
    img = validation(img)
    img = imutils.resize(img, width=300)

    # cv2.imshow('círculos detectados', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('círculos detectados', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3, 3))

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width, channels = cimg.shape
    middle_card = width / 2

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=40, minRadius=20,
                               maxRadius=70)  # parametro2, mientras mas alto mejores circulos encuenra
    circles = circles[0, 0:4]  # toma el circulo olaf pequeño y el grande
    circles = np.uint16(np.around(circles))

    circles2 = sorted(circles, key=lambda x: x[2], reverse=False)  # los ordena
    circles2 = np.uint16(np.around(circles2))
    circles2 = circles2[0:2][0:2]  # tomo el mas pequeño

    if circles2[0][1] > circles2[1][1]:
        larger_y_circle = circles2[0]
        lower_y_circle = circles2[1]
    else:
        larger_y_circle = circles2[1]
        lower_y_circle = circles2[0]



    if circles2[0][0] < middle_card:
        middle = larger_y_circle
        edge = lower_y_circle
    else:
        middle = lower_y_circle
        edge = larger_y_circle

    circles = {path: (middle, edge)}
    #print(circles)
    median = count_circle_pixels(circles)
    print(median)

    # print(i)
    # Dibuja la circusnferencia del círculo
    cv2.circle(cimg, (middle[0], middle[1]), middle[2], (0, 255, 0), 2)  # i[2] radio
    cv2.circle(cimg, (edge[0], edge[1]), edge[2], (0, 255, 0), 2)  # i[2] radio
    cv2.imshow('círculos detectados', cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # dibuja el centro del círculo
    # cv2.circle(cimg, (middle[0], middle[1]), 2, (0, 0, 255), 3)

    mask = np.full((cimg.shape[0], cimg.shape[1]), 0, dtype=np.uint8)
    cv2.circle(mask, (middle[0], middle[1]), middle[2], (255, 255, 255), -1)
    # get first masked value (foreground)
    fg = cv2.bitwise_or(cimg, cimg, mask=mask)

    # get second masked value (background) mask must be inverted
    mask = cv2.bitwise_not(mask)
    background = np.full(cimg.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)

    # combine foreground+background
    final = cv2.bitwise_or(fg, bk)
    cv2.imwrite(os.path.join(folder_perspectives, filename), final)
