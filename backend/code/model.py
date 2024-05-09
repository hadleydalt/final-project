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


def pathToArray(mPaths):
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
        curr_y_train.fill(n)

        x.extend(curr_x_train)
        y.extend(curr_y_train)
    #x = np.expand_dims(x, axis =3)
    return (x, y)

x_train, y_train = pathToArray(paths)
x_test, y_test =  pathToArray(test_paths)

#test_image = io.imread("./s0001_00002_0_0_0_0_0_01.png")
#print(np.shape(test_image))

print(np.shape(x_train))

print("starting model creation")

n_features = np.shape(x_train)[1]
print(n_features)
model = keras.Sequential()
model.add(Dense(10, activation = 'relu', kernel_initializer = 'he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(Dense(1, activation = 'sigmoid'))

print("compiling")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("x_train shape is ")
print(np.shape(x_train))
print(x_train[0])
print("y_train shape is")
print(np.shape(y_train))

print("fitting")
model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=0)

print("2")


#Following this model: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
#I haven't done too much here, but I've started and I feel decent about this direciton if we want to use tensorflow for our modeling