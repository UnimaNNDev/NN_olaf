# Standard imports
import cv2
from math import sqrt
from skimage import data
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import pandas as pd


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

folder = r'D:\UNIMA\CNN_Bi_Class\dataset_limpios_rectangles\positivos'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\CNN_Bi_Class\dataset_limpios_rectangles\positivos'
os.makedirs(folder_perspectives) if not os.path.isdir(folder_perspectives) else None
i = 1
value_blob = []
for path, filename in zip(paths, filenames):
    name = path
    image = cv2.imread(name)
    #_, mask = mask_olaf(image)
    #cv2.imshow('ima', image)
    #image[mask == 0] = 20
    image_gray = rgb2gray(image)
    image_gray_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    #cv2.imshow('ima', image_gray_blur)
    #cv2.waitKey(0)

    blobs_log = blob_log(image_gray_blur, max_sigma=1, num_sigma=10, threshold=.000001)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray_blur, max_sigma=1, threshold=.0001)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray_blur, max_sigma=1.1, threshold=.000001)

    print(len(blobs_doh))
    value_blob.append(len(blobs_doh))

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()

    dict = {'pix_filtered': value_blob}
    df = pd.DataFrame(dict)

    df.to_csv('blobs_positivos.csv')
