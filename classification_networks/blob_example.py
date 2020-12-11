# Standard imports
import cv2
import os
import numpy as np
import random
# Read image

def validation(image):
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def hough_circles(image):
    gray_hough = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_hough, cv2.HOUGH_GRADIENT, 1, 5,
                               param1=50, param2=80, minRadius=60, maxRadius=120)
    flip = False
    try:
        for i in circles[0, :]:
            #print(i[0],i[1])
            if i[0] < img.shape[0]/2:
                flipVertical = cv2.rotate(image, cv2.ROTATE_180)
                #cv2.circle(image, (i[0], i[1]), int(i[2]), (255, 2555, 255), 0)
                #print("hola")
                #cv2.waitKey(0)
                flip = True
            if flip == True:
                break

        return flipVertical
    except:
        return image
    #return image

    #except:
      #  return image

def get_ROI(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX = cX - 3
    cY = cY + 45
    # put text and highlight the center
    #cv2.circle(hough, (cX, cY), 5, (255, 255, 255), -1)
    #cv2.putText(hough, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.rectangle(hough, (cX-20, cY-35), (cX+20, cY+35), (255,0,0), 2)
    ROI = image[cY - 30:cY + 30, cX - 15:cX + 15]
    return ROI
    # display the image
    #cv2.imshow("Image", hough)
    #cv2.waitKey(0)

folder = r'D:\UNIMA\CNN_Bi_Class\positivos_negativos_limpios\train\positivos'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\CNN_Bi_Class\dataset_limpios_rectangles\positivos'
os.makedirs(folder_perspectives) if not os.path.isdir(folder_perspectives) else None
i = 1
for path, filename in zip(paths,
                              filenames):
    #img = cv2.imread(r"C:\Users\Usuario\Desktop\Trabajos\UNIMA\26_11\recortes_parte2\recortes_positivos\dataset_positives_vs_negatives\filtro\train\c\2020-10-31 12_10_54.418608_MIDDLE.png")
    try:
        img = cv2.imread(path)
    except:
        continue
    #!/usr/bin/python
    #cv2.imshow("ori",img)
    vert = validation(img)
    #cv2.imshow("vert",vert)q
    hough = hough_circles(vert)
    #cv2.imshow("hough",hough)
    # convert image to grayscale image
    centroide = get_ROI(hough)

    cv2.imwrite(os.path.join(folder_perspectives, 'centroide_positivo' + str(i) + '.jpg'), centroide)
    i += 1

    #cv2.imshow("",centroide)
    #cv2.waitKey(0)



