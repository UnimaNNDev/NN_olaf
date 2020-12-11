import tensorflow as tf
import cv2
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import os
import numpy as np

def validation(image):
    i = 0
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        i = 1
    return image, i



folder = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\imagenes\tarjetas_olaf_recortados\input'
filenames = os.listdir(folder)
paths = [os.path.join(folder, filename) for filename in filenames]
folder_perspectives = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\imagenes\tarjetas_olaf_recortados\output\output_11_12_limpios'
os.makedirs(folder_perspectives) if not os.path.isdir(folder_perspectives) else None

# These are set to the default names from exported models, update as needed.
model = r'D:\UNIMA\keras-retinanet\modelos\model_nuevos_olaf_17_nov.pb'
#model = r'C:\Users\Usuario\Desktop\custom_Retina\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-\snapshots\model_olaf_v4.pb'
labels_to_names = {0: 'olaf', 1: 'tarjeta'}  ## replace with your model labels and its index value
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

boxes = 'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0'
scores = 'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0'
labels = 'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0'
input_node = 'input_1:0'

with tf.compat.v1.Session() as sess:
    # Coloca como tensor cada una de las salidas
    prob_tensor_boxes = sess.graph.get_tensor_by_name(boxes)
    prob_tensor_scores = sess.graph.get_tensor_by_name(scores)
    prob_tensor_labels = sess.graph.get_tensor_by_name(labels)

with tf.compat.v1.Session() as sess: #creas un device, tensor
    for path, filename in zip(paths,
                              filenames):  # inte en el primer elemento de path y en el primer elemento de filename\n",
        # image_path = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\input\input_10.JPG'  ## replace with input image path
        # output_path = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\output\output_10.png'   ## replace with output image path
        # output_path_edge = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\output\output_10_edge.png'   ## replace with output image path
        # output_path_middle = r'D:\UNIMA\keras-retinanet\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection--master\output\output_10_middle.png'   ## replace with output image path
        card_score = 0
        score_olaf = []
        b_olaf = []
        # load image
        print(path)
        image = cv2.imread(path)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # image = np.expand_dims(image, axis=0)

        # Import the TF graph


        # Realiza las predicciones de la imagen: augmented_image
        predictions_boxes = sess.run(prob_tensor_boxes, {input_node: [image]})
        predictions_scores = sess.run(prob_tensor_scores, {input_node: [image]})
        predictions_labels = sess.run(prob_tensor_labels, {input_node: [image]})

        # correct for image scale
        predictions_boxes /= scale

        # visualize detections
        for box, score, label in zip(predictions_boxes[0], predictions_scores[0], predictions_labels[0]):
            # scores are sorted so we can break
            if score < 0.4:
                break

            color = label_color(label)
            b = box.astype(int)

            # if label == 1:
            #   crop_img = draw[b[1]:b[3], b[0]:b[2]]
            if label == 0:
                if score > card_score:  # deja la tarjeta con el score más alto
                    b_card = b
                    card_score = score
            if label == 1:
                score_olaf.append(score)
                for element in b:
                    b_olaf.append(element)

            #draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        card = draw[b_card[1]:b_card[3], b_card[0]:b_card[2]]
        card, i = validation(card)
        height, width, channels = card.shape


        # te quedas con los dos olaf con score más alta
        try:
            score_olaf_sorted = sorted(score_olaf, reverse=True)
            box_olaf_1 = np.where(score_olaf == score_olaf_sorted[0])  # ubicacion del primer olaf
            box_olaf_2 = np.where(score_olaf == score_olaf_sorted[1])  # ubicacion del segundo olaf
            b_olaf_1 = b_olaf[box_olaf_1[0][0] * 4:box_olaf_1[0][0] * 4 + 4]
            b_olaf_2 = b_olaf[box_olaf_2[0][0] * 4:box_olaf_2[0][0] * 4 + 4]
        except:
            print("No se encontraron todos los olaf :(")

        if i == 0:
            middle_card = (height / 2) + b_card[1]

            # b_olaf_1 = b_olaf[0:4]
            # b_olaf_2 = b_olaf[4:8]

            if b_olaf_1[1] < b_olaf_2[1]:
                higher_y_olaf = b_olaf_2
                lower_y_olaf = b_olaf_1
            else:
                higher_y_olaf = b_olaf_1
                lower_y_olaf = b_olaf_2

            if (((b_olaf_1[3] - b_olaf_1[1]) / 2) + b_olaf_1[1]) > middle_card:  # olaf en la parte inferior
                middle = lower_y_olaf
                edge = higher_y_olaf
            else:
                edge = lower_y_olaf
                middle = higher_y_olaf

            crop_middle = draw[middle[1]:middle[3], middle[0]:middle[2]]
            crop_edge = draw[edge[1]:edge[3], edge[0]:edge[2]]
        else:
            middle_card = (height / 2) + b_card[0]

            # b_olaf_1 = b_olaf[0:4]
            # b_olaf_2 = b_olaf[4:8]

            if b_olaf_1[0] < b_olaf_2[0]:
                higher_y_olaf = b_olaf_2
                lower_y_olaf = b_olaf_1
            else:
                higher_y_olaf = b_olaf_1
                lower_y_olaf = b_olaf_2

            if (((b_olaf_1[2] - b_olaf_1[0]) / 2) + b_olaf_1[0]) > middle_card:  # olaf en la parte inferior
                middle = lower_y_olaf
                edge = higher_y_olaf
            else:
                edge = lower_y_olaf
                middle = higher_y_olaf

        crop_middle = draw[middle[1]-20:middle[3]+20, middle[0]-20:middle[2]+20]
        crop_edge = draw[edge[1]-20:edge[3]+20, edge[0]-20:edge[2]+20]

        detected_img = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)

        nombre_archivo, extension = os.path.splitext(filename)
        try:
            cv2.imwrite(os.path.join(folder_perspectives, nombre_archivo + "_OUTPUT" + extension), card)
            cv2.imwrite(os.path.join(folder_perspectives, nombre_archivo + "_MIDDLE" + extension), crop_middle)
            cv2.imwrite(os.path.join(folder_perspectives, nombre_archivo + "_EDGE" + extension), crop_edge)
        except:
            continue