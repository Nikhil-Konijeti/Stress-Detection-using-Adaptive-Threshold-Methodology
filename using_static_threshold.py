 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import cv2
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random

#Initializing the negative weights
angry_weight=1
fear_weight=1
disgust_weight=1
sad_weight=1

#Initializing the positive weights
happy_weight=-1
surprise_weight=-1

Capture_per_minute=250
neutral_weight=random.uniform(-0.5, 0.5)
Threshold=1000

total_stress=0
window_size=0
iterations=1

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
## End section

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
model = load_model('keras_model/model_5-49-0.62.hdf5')
model.get_config()

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
expressions=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
font = cv2.FONT_HERSHEY_SIMPLEX

data=[]
t_max=[]
total_count=0;
terms_count={}
terms_count['angry'] = 0
terms_count['sad'] = 0
terms_count['disgust'] = 0
terms_count['fear'] = 0
terms_count['happy'] = 0
terms_count['surprise'] = 0

start_time=time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = frame[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
        result = target[np.argmax(model.predict(face_crop))]
        if(result!='neutral'):
            data.append(result)
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)
        if(len(data)>=Capture_per_minute):
           
           stress=0
           stress+=(data.count('angry')*(angry_weight)+
                         data.count('fear')*(fear_weight)+
                         data.count('sad')*(sad_weight)
                         +data.count('disgust')*(disgust_weight)+
                         data.count('happy')*(happy_weight)+
                         data.count('surprise')*(surprise_weight))
           data=[]          
           total_stress+=stress
        if(total_stress<0):
                total_stress=0
        elif(total_stress>Threshold):

            print("Have change")
            
            total_stress=0

    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()