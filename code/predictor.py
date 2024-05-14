import skvideo
import os
from skimage import io
import skvideo.io
import glob
from facefinder import return_eyes
import numpy as np
from model import DDModel
from preprocess import Datasets
from skimage.transform import resize
import tensorflow as tf
import hyperparameters as hp
import cv2


model = DDModel()
model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
model.built = True
model.load_weights("./checkpoints/DD_model/weighty.h5")

frame_path = "../temp_frame_storage/"
training_img_path = "../dataset/USED_DATA/train/Closed_Eyes/"
mean_and_std_path = "../mean_and_std/"


def mean_and_std(path):
    img_list = []
    for filename in os.listdir(path):
        cur_img = io.imread(training_img_path + filename)
        cur_img = np.stack([cur_img, cur_img, cur_img], axis=-1)
        cur_img = resize(cur_img,(hp.img_size, hp.img_size, 3), preserve_range=True)
        
        img_list.append(cur_img/255.)

    mean = np.zeros((hp.img_size,hp.img_size,3))
    std = np.ones((hp.img_size,hp.img_size,3))

    print(np.shape(img_list))
    img_list = np.array(img_list)
    print(np.shape(img_list[0]))

    for channel in range(3):
        std[:, :, channel] = np.std(img_list[:, :, :, channel], axis=0)
        mean[:, :, channel] = np.mean(img_list[:, :, :, channel], axis=0)

    a_mean = mean.reshape(mean.shape[0], -1)
    a_std = std.reshape(std.shape[0], -1)
    np.savetxt(mean_and_std_path + "mean.txt", a_mean)
    np.savetxt(mean_and_std_path + "std.txt", a_std)

    return mean, std

def load_mean_and_std():
    mean = np.loadtxt(mean_and_std_path + "mean.txt")
    l_mean = mean.reshape(hp.img_size,hp.img_size,3)

    std = np.loadtxt(mean_and_std_path + "std.txt")
    l_std = std.reshape(hp.img_size,hp.img_size,3)

    return l_mean, l_std
    

def calc_prediction(path):
    print('starting')

    eye_1_arr = []

    #path = "static" + os.sep + path

    #Loads the video as an array of images. in the future we should limit the length of video here
    video_arr = skvideo.io.vread(path)
    print(np.shape(video_arr))
    

    if(7 > len(video_arr)-7):           #Replace with error message
        print("VIDEO IS TOO SHORT!")

    for i in range(0,len(video_arr)):
        image = video_arr[i]
        face, eye_1 = return_eyes(image)
        #if(eye_1 != None):
        eye_1_arr.append(resize(eye_1,(hp.img_size, hp.img_size, 3), preserve_range=True))
            #eye_2_arr.append(resize(eye_2,(hp.img_size, hp.img_size, 3), preserve_range=True))
        print(i)

    mean, std = load_mean_and_std()

    eye_1_arr = preprocess_image_set(eye_1_arr, mean, std)
    #eye_2_arr = preprocess_image_set(eye_2_arr)
    print("predicting eye_1")
    output_arr = model.predict(eye_1_arr)
    print("predicting eye_2")
    #print(eye_1_arr)
    for i in range(len(output_arr)):
        print(str(i) + " is equal to " + str(np.round(output_arr[i])))
    
    print("ending")
    #print(std)
    
    time = len(video_arr)/30.
    return blink_counter(output_arr), time 


def preprocess_image_set(data, mean, std):

    data = np.array(data, dtype=np.float32)
    data /=  255.

    mean = np.zeros((hp.img_size,hp.img_size,3))
    std = np.ones((hp.img_size,hp.img_size,3))

    for channel in range(3):
        mean[:, :, channel] = np.mean(data[:, :, :, channel], axis=0)
        std[:, :, channel] = np.std(data[:, :, :, channel])


    data = (data-mean) / std
    return data

#1 is open, 0 is closed
def blink_counter(data):
    predict_arr = np.zeros(len(data))
    blink_counter = 0
    blink_switch = True

    for i in range(len(data)):
        if(i > 13):                             #this is for getting N-7, N+7, and N
            before_predict = predict_arr[i-14]
            middle_predict = predict_arr[i-7]
            after_predict = predict_arr[i]
            
            if(before_predict + middle_predict + after_predict) < 1:
                print("check for blink")
                if(blink_switch == True):
                    blink_counter += 1
                    blink_switch = False
            else:
                blink_switch = True
    return blink_counter

#blink, time = calc_prediction("./testing/3_blinks.MOV")
#print("blink is ", str(blink))