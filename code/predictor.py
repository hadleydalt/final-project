import skvideo
import os
import skvideo.io
import glob
from facefinder import return_eyes
from preprocess import vid_preprocess
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

def calc_prediction(path):
    print('starting')

    eye_1_arr = []
    eye_2_arr = []

    new_path = path  #"static/" + path
    #Loads the video as an array of images. in the future we should limit the length of video here
    video_arr = skvideo.io.vread(new_path)
    processed_video = preprocess_image_set(video_arr)

    existing_image_files = glob.glob(frame_path + "*")
    #for f in existing_image_files:
    #    os.remove(f)

    #vidcap = cv2.VideoCapture(path)
    #success,image = vidcap.read()
    #count = 0
    #while success:
     #   cv2.imwrite(frame_path + "frame%d.jpg" % count, image)     # save frame as JPEG file      
      #  success,image = vidcap.read()
       # print('Read a new frame: ', success)
        #count += 1


    predict_arr = np.zeros(len(video_arr))
    blink_counter = 0
    blink_switch = True
    drowsy_threshold = 2
    print(np.shape(video_arr))
    

    if(7 > len(video_arr)-7):           #Replace with error message
        print("VIDEO IS TOO SHORT!")
    for i in range(0,len(video_arr)):
        image = video_arr[i]
        face, eye_1, eye_2 = return_eyes(image)
        if(face != None):
            eye_1_arr.append(eye_1)
            eye_2_arr.append(eye_2)

    existing_image_files_1 = glob.glob(frame_path + "eye_1/" + "*")
    existing_image_files_2 = glob.glob(frame_path + "eye_2" + "*")
    
    for f in existing_image_files_1:
        os.remove(f)
    for f in existing_image_files_2:
        os.remove(f)

    for j in range(0,len(eye_1_arr)):
        cv2.imwrite(frame_path + "eye_1/" "frame%d.jpg" % j, eye_1_arr[j])     # save frame as JPEG file  
        cv2.imwrite(frame_path + "eye_2/" + "frame%d.jpg" % j, eye_1_arr[j])     # save frame as JPEG file      
        print('Read a new frame: ', j)
    
    eye_1_arr = preprocess_image_set(eye_1_arr)
    eye_2_arr = preprocess_image_set(eye_2_arr)
    print("predicting eye_1")
    print(model.predict(eye_1_arr))
    print("predicting eye_2")
    
    time =2
    print(time)
    print("ending")
    return blink_counter, time 


def preprocess_image_set(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    data = np.divide(data, 255)
    data = (data/mean) / std
    return data

def get_eye_predict(image):
    '''
    So based on get_eye_predict, there are no confidence scores? It's just a drowsiness percentage? 
    Because if so, can't we average together the percentages and return that to the frontend 
    '''
    #this is a dummy function. when we have our model done, replace this or any
    #calls to this function with the function to pass in an image and pass back out its classification
    #My base assumptions here: 1 is eyes closed, 0 is eyes open. I understand we're going to get a percentage here.


    #y = model(image, training=False)
    #print("Result is ")
    #print(y)
    #print("~~~~~~~~~~~~~~~~~~~~~~~~")

    return 87

calc_prediction("./testing/pogBlink.MOV")

'''
            if(i > 13):                             #this is for getting N-7, N+7, and N
                before_predict = predict_arr[i-14]
                middle_predict = predict_arr[i-7]
                after_predict = predict_arr[i]
                
                if(before_predict + middle_predict + after_predict) > drowsy_threshold:
                    if(blink_switch):
                        blink_counter += 1
                        blink_switch = False
                else:
                    blink_switch = True
                    '''