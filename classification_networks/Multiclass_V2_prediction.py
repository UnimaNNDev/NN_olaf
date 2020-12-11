import numpy as np # linear algebra
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import tensorflow as tf
from sklearn.metrics import classification_report


# CARGAR MODELO ## :)
model = tf.keras.models.load_model(r'D:\UNIMA\CNN_Bi_Class\multicalse_v2_modelos\multiclass_v2_positive_negative_pearls.h5')
images = []
true = []
img_folder = os.path.join(r'D:\UNIMA\CNN_Bi_Class\test_olaf')
img_files = os.listdir(img_folder)
img_files = [os.path.join(img_folder, f) for f in img_files]
# print(img_files)
for img in img_files:
    name = img
    print(img)
    img = load_img(img, target_size=(150, 150)) #150, 224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

    if "neg" in name:
        true_label = 0
    elif "pearlas" in name:
        true_label = 1
    elif "pos" in name:
        true_label = 2

    true.append(true_label)

# stack up images list to pass for prediction
images = np.vstack(images)
# print(images)
classes = model.predict_classes(images, batch_size=10)

print(classification_report(true, classes))

print(classes)