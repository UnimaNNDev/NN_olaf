import tensorflow as tf
import os
import numpy as np
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    #img = img - [123.68, 116.779, 103.939]
    return img


folder = r'D:\UNIMA\CNN_Bi_Class\test_olaf'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\imagenes\tarjetas_olaf_recortados\output\positivos_negativos_prueba'
os.makedirs(folder_perspectives) if not os.path.isdir(folder_perspectives) else None

# These are set to the default names from exported models, update as needed.
model = r'D:\UNIMA\CNN_Bi_Class\modelo_pb_CNN_python\frozen_final_12_nov.pb'
#model = r"D:\UNIMA\tensorflow-yolov4-tflite\checkpoints\yolov4-416\saved_model.pb"
graph_def = tf.compat.v1.GraphDef()

with tf.io.gfile.GFile(model, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)

# These names are part of the model and cannot be changed.
# You need to change the outputs for your model

output_node = "Identity:0"
input_node = 'x:0'

with tf.compat.v1.Session() as sess:
    # Coloca como tensor cada una de las salidas
    prob_tensor_output = sess.graph.get_tensor_by_name(output_node)

for path, filename in zip(paths,
                          filenames):  # inte en el primer elemento de path y en el primer elemento de filename\n",


    # load image
    #image = cv2.imread(path)
    image = load_image(path)

    #dim = (224, 224)
    # resize image
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Import the TF graph

    with tf.compat.v1.Session() as sess:
        # Realiza las predicciones de la imagen: augmented_image
        predictions = sess.run(prob_tensor_output, {input_node: [image]})
        print(filename, predictions[0])
