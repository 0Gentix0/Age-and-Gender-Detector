import cv2
import sys
from tensorflow import keras
import numpy as np
ageDetector = keras.models.load_model('agedetector.h5')

genderDetector = keras.models.load_model('genderdetector.h5')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
i=0
gender=["male","female"]
age=["<19","19-35","35-60","60<"]
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if i==40:
            image = cv2.resize(gray[x:x+h,y:y+h],dsize=(200,200))
            images=np.zeros((1,200,200,1),dtype = 'float32')
            images[0,:,:,0]=image
            print(age[np.argmax(ageDetector(images).numpy())],end=" ")
            print(gender[np.argmax(genderDetector(images).numpy())],end=" ")
    if i==40:
        print()
        i=0
    # Display the resulting frame
    cv2.imshow('Video', frame)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()