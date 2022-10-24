import glob
import os
import imagesize
import numpy as np
import pandas as pd
import tensorflow as tf


def explore(path):
    dir_classes = dict()

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            dir_classes[subdirname] = os.path.join(dirname, subdirname)
            # print(os.path.join(dirname, subdirname)

    return dir_classes


def view_shapes(path_img):
    return imagesize.get(path_img)


class Preprocessing():
    def __init__(self) -> None:
        pass

    def read_and_decode(self, filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        return img

    def preprocess_image(self, img, width=64, height=64,to_bytes=True):
        if (img.shape[0] < width and img.shape[1] < height):
            img = tf.image.resize(img, [width, height],
                                  preserve_aspect_ratio=True)
        if img.shape[0] > width:
            img = tf.image.resize(img, [int(width*1.3), int(height*1.3)],
                                  preserve_aspect_ratio=True)
        img = tf.image.resize_with_crop_or_pad(img,
                                               target_width=width,
                                               target_height=height)
        if to_bytes:
            img=tf.cast(img,tf.uint8)
            img=tf.image.encode_jpeg(img).numpy()
            return img
       
        return img.numpy().tolist()


