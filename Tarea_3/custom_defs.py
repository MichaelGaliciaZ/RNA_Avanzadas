import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
import imagesize 

def explore(path):
    dir_classes = dict()

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            dir_classes[subdirname] = os.path.join(dirname, subdirname)
            # print(os.path.join(dirname, subdirname)

    return dir_classes

def view_shapes(path_img):
    return imagesize.get(path_img)

def preprocess_image(img):
    pass


def preprocess_batch(img):
    pass


def export_dataset(format):
    pass


def export_batch_dataset(format):
    pass

