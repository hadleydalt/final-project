import cv2
import matplotlib.pyplot as plt
import numpy as np

#Takes in an input image (NOT A PATH, but we can discuss this) and returns the face and eyes in (x, y, width, height) format for each
def return_eyes(input_image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')  #for finding the eyes
    f_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  #for finding the face faster

    #return lists
    face_list = []
    eye_list = []

    #converts our image to rgb, which dlib likes more
    c_image = input_image
    d_faces = f_detector.detectMultiScale(c_image, minSize=(60,60))

    for (x,y,w,h) in d_faces: #theoretically this supports multiple faces, but we only really want to detect one. we should have a system in place for 1) too many faces and 2) no faces

        face_list.append([x,y,w,h])
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cuts out just the face for eye detection
        j_face = input_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(j_face, minSize=(60,60))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(input_image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255,0,0), 2)
            eye_list.append(j_face[ey:ey+eh, ex:ex+ew])

    plt.imshow(input_image, cmap='gray')
    plt.axis('off')
    plt.show()
    if  (len(face_list) < 1) or (len(eye_list) < 2):
        return None, None
    return face_list[0], eye_list[0]