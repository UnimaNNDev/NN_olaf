import imutils
import cv2
import os

inter = cv2.INTER_AREA


folder = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\imagenes\tarjetas_olaf_recortados\output\positivos_negativos_prueba'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\imagenes\tarjetas_olaf_recortados\output\prueba'


i=0
for path, filename in zip(paths, filenames):  # inte en el primer elemento de path y en el primer elemento de filename\n",

    image = cv2.imread(path)
    ## CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    # convert image from RGB to HSV
    #img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ## Histogram equalisation on the V-channel
    #img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    ## convert image back from HSV to RGB
    #image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)




    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)


    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # _____END_____#
    name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(folder_perspectives, name +".jpg"), final)
    ##cv2.imwrite(os.path.join(folder_perspectives, "positive." + str(i) +".jpg"), image)
    i=i+1
