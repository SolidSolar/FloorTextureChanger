import os
import test
import ml
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from vanishing_point import hough_transform, find_intersections, sample_lines, find_vanishing_point
import tensorflow_datasets as tfds

black = 0
white = 255

def load_image_test(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224), method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(
        img, (1, 224, 224, 3), name = None
    )
    return img

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def change_floor(image, imagecv2, floor_texture):
    mask = np.array(tf.keras.utils.array_to_img(test.get_mask(image)[0]))

    mask = mask[:, :, ::-1].copy()
    imagecv2 = cv2.resize(imagecv2, (224, 224))
    buffer = np.copy(imagecv2)
    hough_lines = hough_transform(imagecv2)
    thresh = 127
    im = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im, 224, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if hough_lines:
        for line in hough_lines:
            cv2.line(imagecv2, line[0], line[1], (255, 0, 0), 1)
        random_sample = sample_lines(hough_lines, 100)
        intersections = find_intersections(random_sample)
        tmpCollection = intersections.copy()
        for i in tmpCollection:
            if i[0] < 0 or i[1] < 0 or i[0] > imagecv2.shape[0] or i[1] > imagecv2.shape[1]:
                continue
            if im_bw[int(i[0]), int(i[1])] != black:
                intersections.remove(i)

        if intersections:
            grid_size = min(imagecv2.shape[0], imagecv2.shape[1]) // 6
            vanishing_point = find_vanishing_point(buffer, grid_size, intersections)
            result = ml.render(buffer, floor_texture, vanishing_point, im_bw)
            cv2.imwrite('result.jpg', result)
            return 0
        else:
            return -1

def display(display_list):
 plt.figure(figsize=(15, 15))

 title = ["Input Image", "True Mask", "Predicted Mask"]

 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()

def create_mask(pred_mask):
 pred_mask = tf.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask[0]

def show_predictions(dataset, num=10):
    for image, mask in dataset.take(num):
        pred_mask = unet_model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])

def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (224, 224), method="nearest")
   input_mask = tf.image.resize(input_mask, (224, 224), method="nearest")
   return input_image, input_mask

def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask


def load_image_test1(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)

   return input_image, input_mask

dataset, info = tfds.load('my_dataset', with_info=True)
test_dataset = dataset["test"].map(load_image_test1, num_parallel_calls=tf.data.AUTOTUNE)
unet_model = tf.keras.models.load_model('model')
imagecv2 = cv2.imread('room.jpg')
image = load_image_test('room.jpg')
tile = cv2.imread('tile.jpg')
change_floor(image, imagecv2, tile)
