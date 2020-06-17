from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from detect import *


face_classifier = cv2.CascadeClassifier("/Users/brenda/Downloads/IS_Final_Project/haarcascade_frontalface_default.xml")
classifier =load_model("/Users/brenda/Downloads/IS_Final_Project/Emotions_vgg.h5")

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# def face_detector(img):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)
#     if faces is ():
#         return (0,0,0,0),np.zeros((48,48),np.uint8),img

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h,x:x+w]

#     try:
#         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
#     except:
#         return (x,w,y,h),np.zeros((48,48),np.uint8),img
#     return (x,w,y,h),roi_gray,img

cap = cv2.VideoCapture(0)


def emotion_detection():
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            #drawing the frame
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,153,153),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)


            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class
                prediction = classifier.predict(roi)
                preds = classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                percentage = preds.item(preds.argmax())
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(204,255,153),2)
                cv2.putText(frame, str(percentage), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 2)

            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def run_emotion_detection():
    emotion_detection()
    run()

def run():
    menu = "[1] Emotion Detection" \
           "\n[2] Gender & Age Detection" \
           "\n[3] Exit" \
           "\nPlease choose 1/2/3"
    print(menu)
    choice = input("Enter your choice: ")
    if choice == "1":
        run_emotion_detection()

    elif choice == "2":
        gender_age_detection()
        run()
    elif choice=="3":
        print("Goodbye!")
    else:
        print("Please choose from the options!")
        run()


run()























