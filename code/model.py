import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt
import glob
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.regularizers import L2, L1

import hyperparameters as hp


#i'm saying 0 is eyes closed, 1 is eyes open

class DDModel(tf.keras.Model):
    """neural network model drowsiness-detection. """

    def __init__(self):
        super(DDModel, self).__init__()

        # TASK 1
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)
        
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = hp.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
            )

       # added the dropout layer after the last pooling layer based on this 
       # reference post I read through 
       # https://saturncloud.io/blog/where-to-add-dropout-in-neural-network/#:~:text=For%20example%2C%20if%20you%20have,the%20size%20of%20your%20dataset.

        self.architecture = [
            Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.1),

            Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.2),


            Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.2),

            Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.25),

            Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.3),


            Flatten(),

            Dense(256,activation='relu'),
            Dropout(0.4),
            Dense(128,activation='relu'),
            Dropout(0.4),
            Dense(15),
            Dense(1, activation="sigmoid")
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        # TASK 1
        # TODO: Select a loss function for your network 
        #       (see the documentation for tf.keras.losses)

        #  Tried using sprse categorical cross entropy. For two or more label classes
        loss_function = tf.keras.losses.BinaryCrossentropy()
        return loss_function(labels,predictions)