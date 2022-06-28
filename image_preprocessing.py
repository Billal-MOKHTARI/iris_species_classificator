from tkinter.tix import AUTO
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(directory, size, val_split, shuffle, subset, seed, autotune=False):
    data = keras.preprocessing.image_dataset_from_directory(
        directory = directory,
        batch_size=32,
        image_size=size,
        shuffle=shuffle,
        seed=seed,
        validation_split=val_split,
        subset=subset,
    )
    if autotune == True:
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = data.prefetch(buffer_size=AUTOTUNE)
    return data

def show_img(dataset, figsize, nx, ny):
    class_names = dataset.class_names

    plt.figure(figsize=figsize, )
    for images, labels in dataset.take(1):
        for i in range(nx*ny):
            ax = plt.subplot(nx, ny, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

def data_augmenter(augmenter_list):
    data_augmentation = keras.Sequential()
    for i in range(len(augmenter_list)):
        data_augmentation.add(augmenter_list[i])

    return data_augmentation
    
def data_rescaling(value):
    rescale = keras.Sequential(
        layers.Rescaling(value)
    )

    return rescale
