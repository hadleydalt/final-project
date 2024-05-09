import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt
import glob


import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.regularizers import L2, L1

import hyperparameters as hp



print(tf.__version__)


#i'm saying 0 is eyes closed, 1 is eyes open

#paths = ["C:/Users/shnur/OneDrive/Documents/Datasets/Just eyes - Prasad drowsiness detection dataset/train/Closed_Eyes/", "C:/Users/shnur/OneDrive/Documents/Datasets/Just eyes - Prasad drowsiness detection dataset/train/Open_Eyes/"]
#test_paths = ["C:/Users/shnur/OneDrive/Documents/Datasets/eye_test - Elilah/close Eyes/","C:/Users/shnur/OneDrive/Documents/Datasets/eye_test - Elilah/open Eyes/"]
#types = ["eyes_closed", "eyes_open"]


def pathToArray(mPaths, mTypes):
    x=[]
    y=[]
    for n in range(len(mPaths)):
        path = mPaths[n]

        #get list of paths
        img_name_list = glob.glob(path + "*.png")

        #loads the files, resizes them to 64 x 64, and then creates a whole array for them
        curr_x_train = np.array([np.array(cv2.resize(io.imread(img_name), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)) for img_name in img_name_list])

        #get list of equal length, set their value to 0 or 1 accordingly as their tag
        curr_y_train = np.empty((len(curr_x_train),1))   
        curr_y_train.fill(mTypes[n])

        x.extend(curr_x_train)
        y.extend(curr_y_train)
    #x = np.expand_dims(x, axis =3)
    return (x, y)
def trainModel(curr_paths, curr_test_paths):
    x_train, y_train = pathToArray(curr_paths,types)
    x_test, y_test =  pathToArray(curr_test_paths,types)

    x_train = np.reshape(x_train, (np.shape(x_train)[0], np.shape(x_train)[1], np.shape(x_train)[2], 1))
    x_test = np.reshape(x_test, (np.shape(x_test)[0], np.shape(x_test)[1], np.shape(x_test)[2], 1))

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    in_shape = x_train.shape[1:]

    print(x_train.shape)
    print(in_shape)

    print("starting model creation")

    n_features = np.shape(x_train)[1]
    print(n_features)
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(curr_paths), activation='softmax'))

    print("compiling")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("x_train shape is ")
    print(np.shape(x_train))
    print("y_train shape is")
    print(np.shape(y_train))

    print("fitting")
    model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)

    print("2")


#trainModel(paths, test_paths)


#Following this model: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
#I haven't done too much here, but I've started and I feel decent about this direction if we want to use tensorflow for our modeling




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

        # TASK 1
        # TODO: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.


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
            Dense(15)
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
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)
        return loss_function(labels,predictions)
