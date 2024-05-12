import skvideo.io
from facefinder import return_eyes
import numpy as np



def calc_prediction(path):
    #Loads the video as an array of images. in the future we should limit the length of video here
    video_arr = skvideo.io.vread(path)
    predict_arr = np.zeros(len(video_arr))
    drowsy_threshold = 2
    print(np.shape(video_arr))
    

    if(7 > len(video_arr)-7):           #Replace with error message
        print("VIDEO IS TOO SHORT!")
    for i in range(0,len(video_arr)):
        image = video_arr[i]
        face, eye_1, eye_2 = return_eyes(image)
        predict_1 = get_eye_predict(eye_1)
        predict_2 = get_eye_predict(eye_2)

        predict_arr[i] = (predict_1 + predict_2)/2 #not sure how we want to deal with the percentage of one eye closed vs the other here. Do we average them like I'm doing here?

        if(i > 13):                             #this is for getting N-7, N+7, and N
            before_predict = predict_arr[i-14]
            middle_predict = predict_arr[i-7]
            after_predict = predict_arr[i]
            
            if(before_predict + middle_predict + after_predict) > drowsy_threshold:
                return True
        
    '''
    This is a pretty stupid system so far. When we were doing still images it made sense to just return the percentage certainty as a drowsiness percentage,
    but now that we're doing video that system falls apart. One idea here that would make sense is if the N-7/N+7 returns that the persons eyes have been closed for
    awhile, we return that they're drowsy, but that's much more binary than a percentage drowsiness. Not very interesting, we should discuss
    '''

    '''
    QUESTION: What if the person were to just blink 3 times or something because they have something in their eye? Or if they are drowsy but just happen to have their eyes open during the frames in question? 
    '''

    return False


def get_eye_predict(image):
    '''
    So based on get_eye_predict, there are no confidence scores? It's just a drowsiness percentage? 
    Because if so, can't we average together the percentages and return that to the frontend 
    '''
    #this is a dummy function. when we have our model done, replace this or any
    #calls to this function with the function to pass in an image and pass back out its classification
    #My base assumptions here: 1 is eyes closed, 0 is eyes open. I understand we're going to get a percentage here.
    return 87

#calc_prediction("./testing/pog.mov")