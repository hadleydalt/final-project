import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt
import glob
print(tf.__version__)


#i'm saying 0 is eyes closed, 1 is eyes open

paths = ["C:/Users/shnur/OneDrive/Documents/Datasets/Just eyes - Prasad drowsiness detection dataset/train/Closed_Eyes/", "C:/Users/shnur/OneDrive/Documents/Datasets/Just eyes - Prasad drowsiness detection dataset/train/Open_Eyes/"]
test_paths = ["C:/Users/shnur/OneDrive/Documents/Datasets/eye_test - Elilah/close Eyes/","C:/Users/shnur/OneDrive/Documents/Datasets/eye_test - Elilah/open Eyes/"]
types = ["eyes_closed", "eyes_open"]


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


trainModel(paths, test_paths)


#Following this model: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
#I haven't done too much here, but I've started and I feel decent about this direction if we want to use tensorflow for our modeling