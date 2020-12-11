
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report



##prueba


import warnings
import io
import os
import cv2
try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None


def cargar_imagen(path, target_size=None, interpolation='nearest'):
    img = cv2.imread(path, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.resize(224, 224, 3)
    return img


# CARGAR MODELO ## :)
model = tf.keras.models.load_model(r'D:\UNIMA\CNN_Bi_Class\biclass_modelos\model_09_12_pearls_others.h5')
images = []
true = []
img_folder = os.path.join(r'D:\UNIMA\CNN_Bi_Class\test_olaf')
img_files = os.listdir(img_folder)
img_files = [os.path.join(img_folder, f) for f in img_files]
# print(img_files)
for img in img_files:
    name = img
    print(img)
    img = cargar_imagen(img, target_size=(224, 224)) #150, 224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
    #print(images)

    if "others" in name:
        true_label = 0
    elif "pearls" in name:
        true_label = 1

    true.append(true_label)

# stack up images list to pass for prediction
images = np.vstack(images)
# print(images)
classes = model.predict(images, batch_size=10)

predicted = []

for i in classes:
    if i >= 0:
        predicted.append(1)
    else:
        predicted.append(0)


print(predicted)

print(classification_report(true, predicted))

print(classes)