import skvideo.io
from facefinder import return_eyes
import numpy as np



def calc_prediction(path):
    #Loads the video as an array of images. in the future we should limit the length of video here
    video_arr = skvideo.io.vread(path)
    predict_arr = np.zeros(len(video_arr))
    print(np.shape(video_arr))
    

    if(7 > len(video_arr)-7):           #Replace with error message
        print("VIDEO IS TOO SHORT!")
    for i in range(0,len(video_arr)):
        face, eye_1, eye_2 = return_eyes(image)
        predict_1 = get_eye_predict(eye_1)
        predict_2 = get_eye_predict(eye_2)

        predict_arr[i] = (predict_1 + predict_2)/2 #not sure how we want to deal with the percentage of one eye closed vs the other here. Do we average them like I'm doing here?

        if(i > 13):                             #this is for getting N-7, N+7, and N
            before_predict = predict_arr[i-14]
            middle_predict = predict_arr[i-7]
            after_predict = predict_arr[i]
 
    for image in video_arr:
        face, eye_1, eye_2 = return_eyes(image)

        predict_1 = get_eye_predict(eye_1)
        predict_2 = get_eye_predict(eye_2)

    #More to do:
    #- Storing the predictions for each image
    #- calculating the blink threshold
    #- returning the value of how drowsy they are

    return False


def get_eye_predict(image):
    #this is a dummy function. when we have our model done, replace this or any
    #calls to this function with the function to pass in an image and pass back out its classification
    return 87
calc_prediction("./testing/pog.mov")