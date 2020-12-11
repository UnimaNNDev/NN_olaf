import os
import cv2
import numpy as np
import imutils
inter = cv2.INTER_AREA


folder = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\olaf sin ejecutar'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\olaf'


i=0
for path, filename in zip(paths, filenames):  # inte en el primer elemento de path y en el primer elemento de filename\n",

    image = cv2.imread(path)
    #cimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cimage = cv2.blur(cimage, (3, 3))


    #circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=40, minRadius=0, maxRadius=0)  # parametro2, mientras mas alto mejores circulos encuenra
    #circles = np.uint16(np.around(circles))
    #for j in circles[0, :]:
        # draw the center of the circle
     #   cv2.circle(image, (j[0], j[1]), 6, (0, 0, 255), -1)
    #print(folder_perspectives, "negative." + str(i) +".jpg")
    #new_image = cv2.resize(image, (224,224), interpolation = inter)
    #cv2.imwrite(os.path.join(folder_perspectives, "positive." + str(i) +".jpg"), image)
    name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(folder_perspectives, name +".jpg"), image)

    i=i+1
