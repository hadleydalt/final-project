import skvideo
import skvideo.io
from facefinder import return_eyes
import numpy as np
from model import DDModel
from skimage.transform import resize


#model = DDModel()
#model.built = True
#model.load_weights("./testing/your.weights.e009-acc0.9935.h5")
def calc_prediction(path):
    print('starting')
    new_path = path  #"static/" + path
    #Loads the video as an array of images. in the future we should limit the length of video here
    video_arr = skvideo.io.vread(new_path)
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
            predict_1 = get_eye_predict(resize(eye_1, (64, 64)))
            predict_2 = get_eye_predict(resize(eye_2, (64, 64)))

            predict_arr[i] = (predict_1 + predict_2)/2 #not sure how we want to deal with the percentage of one eye closed vs the other here. Do we average them like I'm doing here?

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
    time = len(predict_arr)/24
    print(time)
    print("ending")
    return blink_counter, time 


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

calc_prediction("C:/Users/shnur/OneDrive/Desktop/CS1430/CS1430_Projects/final-project/code/testing/pogBlink.MOV")