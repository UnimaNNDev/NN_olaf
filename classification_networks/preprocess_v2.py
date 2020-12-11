import cv2
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report

def mask_olaf(olaf):
    gray = cv2.cvtColor(olaf, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh1 = cv2.Canny(blur, 70, 120)
    thresh1 = cv2.GaussianBlur(thresh1, (5, 5), 1)

    cnt = sorted(cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

    mask = np.zeros((olaf.shape[0], olaf.shape[1]), dtype=np.uint8)

    cv2.drawContours(mask, [cnt], -1, 255, -1)
    dst = cv2.bitwise_and(olaf, olaf, mask=mask)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    dst_gray[dst_gray > 140] = 0
    dst_erode = cv2.erode(dst_gray, (3, 3), iterations=2)

    # cv2.imshow('edge', dst_erode)
    # cv2.waitKey()
    return (dst_erode, mask)


images = []
img_folder = os.path.join(r'D:\UNIMA\CNN_Bi_Class\positivos_negativos_limpios\train\positivos')
img_files = os.listdir(img_folder)
img_files = [os.path.join(img_folder, f) for f in img_files]
# blob detector
detector = cv2.SimpleBlobDetector_create()
failed = 0
failed_paths = []
amount = 1
claheLAB = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
circle_analysis = defaultdict(lambda: defaultdict(dict))

value_1 = []
value_2 = []
value_3 = []
value_4 = []

positivo = 0
negativo = 0

index = 0
for image in img_files:
    if image == r'D:\UNIMA\CNN_Bi_Class\positivos_negativos_limpios\train\positives\desktop.ini':
      continue
    # print(f'Analizando imagen {image}')
    image = cv2.imread(image)
    original = image.copy()

    if image.shape[0] / image.shape[1] > 1:
        # giramos la imagen con transpose para que todas queden en vertical
        image = cv2.transpose(image)
    
    gray_hough = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray_hough,cv2.HOUGH_GRADIENT,1,100,
                        param1=50,param2=50,minRadius=50,maxRadius=200)

    #try:
     #   for i in circles[0,:]:
      #      #cv2.circle(image,(int(i[0]),int(i[1])),int(i[2]),(0,0,0),-1)
       #     mask = np.full((original.shape[0], original.shape[1]), 0, dtype=np.uint8)
        #    cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
         #   # get first masked value (foreground)
          #  fg = cv2.bitwise_or(original, original, mask=mask)
           # cv2.imshow('',fg)
    #except:
    try:
        for i in circles[0, :]:
            #cv2.circle(image,(int(i[0]),int(i[1])),int(i[2]),(0,0,0),-1)
            mask = np.full((original.shape[0], original.shape[1]), 0, dtype=np.uint8)
            cv2.circle(mask, (int(i[0]),int(i[1])),int(i[2]), (255, 255, 255), -1)
        # get first masked value (foreground)
            fg = cv2.bitwise_or(original, original, mask=mask)
    except:
        continue

     #   continue
    image = cv2.resize(image, (350, 350))
    # cv2.imshow('', image)
    # removing background with hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(result)
    filter = g.copy()
    # masking
    ret, mask = cv2.threshold(filter, 5, 255, 1)
    image[mask == 0] = 255

    ####### processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show_image(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # show_image(lab)
    # yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # show_image(yuv)
    # _, otsu_bin = cv2.threshold(
    #    image_gray, 0, 140, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # show_image(otsu_bin)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(15, 15))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    hsv[..., 1] = clahe.apply(hsv[..., 1])
    hsv[..., 0] = clahe.apply(hsv[..., 0])
    lab[..., 0] = claheLAB.apply(lab[..., 0])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_lab = cv2.cvtColor(bgr, cv2.COLOR_LAB2RGB)
    # _, ordinary_img = cv2.threshold(rgb, 150, 230, cv2.THRESH_BINARY)
    # show_image(rgb, title="rgb from hsv")
    # perspective = intensity_fix(rgb)
    # show_image(perspective)
    # show_image(lab, title="lab")
    # show_image(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), title="from lab")
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # blurred = cv2.GaussianBlur(image, (51, 51), 0)
    blurred_rgbHSV = cv2.GaussianBlur(rgb, (3, 3), 0)
    blurred_rgbLAB = cv2.GaussianBlur(rgb_lab, (3, 3), 0)
    # show_image(blurred, title= 'blurred')
    # show_image(blurred_rgbHSV, title='blurred rgb-HSV')
    # show_image(blurred_rgbLAB, title='blurred RGB-LAB')
    filtered_lab = image - blurred_rgbLAB
    filtered_hsv = image - blurred_rgbHSV
    # show_image(filtered_lab, title= 'filtered lab')
    # show_image(filtered_hsv, title='filtrada hsv')
    filtered1 = image - blurred
    filtered = filtered1 + 127 * np.ones(image.shape, np.uint8)
    filter_corrected_lab = filtered_lab + 127 * np.ones(image.shape, np.uint8)
    filter_corrected_hsv = filtered_hsv + 127 * np.ones(image.shape, np.uint8)

    # cv2.imshow('filtered1', filtered1)
    # cv2.imshow('filtered', filtered)
    # cv2.imshow('corrected lab', filter_corrected_lab)
    # cv2.imshow('corrected hsv', filter_corrected_hsv)
    # print(f'filtered shape: {filter_corrected_lab.shape}')
    #####
    labGray = cv2.cvtColor(rgb_lab, cv2.COLOR_RGB2GRAY)
    hsvGray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    #####
    # gray_perspective = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_perspective = cv2.cvtColor(filtered1, cv2.COLOR_RGB2GRAY)
    gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
    gray_corrected_lab = cv2.cvtColor(filter_corrected_lab, cv2.COLOR_RGB2GRAY)
    gray_corrected_hsv = cv2.cvtColor(filter_corrected_hsv, cv2.COLOR_RGB2GRAY)
    dst_erode, mask = mask_olaf(image)
    gray_perspective[mask == 0] = 20
    gray_filtered[mask == 0] = 20
    gray_corrected_lab[mask == 0] = 20
    gray_corrected_hsv[mask == 0] = 20
    #    gray_perspective[ mask == 0] = 255
    # show_image(gray_perspective, title="gray filtered1")
    # show_image(gray_filtered, title="gray filtered")
    # show_image(gray_corrected_lab, title='gray corrected lab')
    # show_image(gray_corrected_hsv, title="gray corrected hsv")
    ret, perspective_filtered1 = cv2.threshold(gray_perspective, 100, 255, cv2.THRESH_BINARY)
    ret, perspective_filtered = cv2.threshold(gray_filtered, 100, 255, cv2.THRESH_BINARY)
    ret, perspective_lab = cv2.threshold(gray_corrected_lab, 100, 255, cv2.THRESH_BINARY)
    ret, perspective_hsv = cv2.threshold(gray_corrected_hsv, 100, 255, cv2.THRESH_BINARY)

    #cv2.imshow('hoki', dst_erode)
    # cv2.imshow('', perspective_filtered)
    # cv2.imshow('', perspective_lab)
    # cv2.imshow('', perspective_hsv)
    pix_filtered1 = cv2.countNonZero(perspective_filtered1)
    pix_filtered = cv2.countNonZero(perspective_filtered)
    pix_lab = cv2.countNonZero(perspective_lab)
    pix_hsv = cv2.countNonZero(perspective_hsv)
    # print(f'pixeles filtreres 1 {pix_filtered1}')
    # print(f'pixeles filtreres {pix_filtered}')
    # print(f'pixeles filtreres 1ab {pix_lab}')
    # print(f'pixeles filtreres hsv {pix_hsv}')

    value_1.append(pix_filtered)
    value_2.append(pix_filtered1)
    value_3.append(pix_lab)
    value_4.append(pix_hsv)

    keypoints = detector.detect(gray_filtered)
    # print(len(keypoints))
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(original, keypoints, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    #cv2.imshow('', blobs)
    # Negativo 3000 filtered1
    # Positivo 3000- and so on

    if pix_filtered <= 3000:
        dx = "negativo"
        negativo = negativo + 1
    elif pix_filtered > 3000:
        dx = "positivo"
        positivo = positivo + 1

    index += 1

    cv2.waitKey(0)

    #cv2.imwrite(r'D:\UNIMA\CNN_Bi_Class\imagenes_black_hole\positvas'+'\positivo_'+ str(index) + '.jpg', fg)
#print(positivo, negativo)

dict = {'pix_filtered': value_1, 'pix_filtered1': value_2, 'pix_lab': value_3, 'pix_hsv': value_4}
df = pd.DataFrame(dict)

df.to_csv('positivos_04_12_2020.csv')
