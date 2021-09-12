#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random
import time
import numpy as np

#Initializing the negative weights
angry_weight=1
fear_weight=1
disgust_weight=1
sad_weight=1
alpha = 5
#Initializing the positive weights
happy_weight=-1
surprise_weight=-1
#neutral_weight=random.uniform(-0.5, 0.5)

#Initializing the parameters
Capture_per_minute=250
Threshold = 1000
t_old=Threshold
t_new = 0
windows = 0
pre_windows = [] 
total_stress=0
iterations=1
flag = 0

#Loading the pretrained weights
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
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
previous_threshold = []

partially_negative=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','disgust','sad']
partially_positive=['happy', 'sad', 'surprise','happy','surprise']

iter2=[]
stress=[]

i=0
data=[]
tot_str=0
iter2.append(i)
stress.append(tot_str)
u=0
start_time=time.time()
t_store = Threshold
partially_negative=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','disgust','sad']
partially_positive=['happy', 'sad', 'surprise','happy','surprise']

iter2=[]
stress=[]
 
font = cv2.FONT_HERSHEY_SIMPLEX
#Initialize alpha
#Do plotting
#Find proper data for validation


def f(windows, pre_windows, alpha):
        t_max = int(sum(pre_windows)/len(pre_windows))
        t_ite = windows
        t = alpha**(-1*(t_ite/t_max))
        return t
        
start_time=time.time()  
stress1 = 0
while(1):
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
        data.append(result) 
        if(result == 'happy' or result =='surprise'):
                result = "Positive_Emotion"
        elif(result == 'neutral'):
                result = "Neutral"
        else:
                 result = "Negative_Emotion"
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)
    if(len(data)>Capture_per_minute):
        stress1+=(data.count('angry')*(angry_weight)+
                             data.count('fear')*(fear_weight)+
                             data.count('sad')*(sad_weight)
                             +data.count('disgust')*(disgust_weight)+
                             data.count('happy')*(happy_weight)+
                             data.count('surprise')*(surprise_weight))
        tot_str = tot_str+stress1
        data=[]
        print('my tot_str variable',tot_str)
        i=i+1
        #time.sleep(60)
        if(flag ==0):
                if(tot_str<=0):
                    tot_str=0
                    windows  = 0
                    start_time = time.time()
                    stress.append(tot_str)
                    iter2.append(i)
                if(tot_str>0 and tot_str<Threshold):
                    windows = windows+1
                if(tot_str>Threshold):    
                    end_time = time.time()
                    t_new = end_time- start_time
                    print("Have a change")
                    tot_str=0
                    flag = 1
                    previous_threshold.append(Threshold)
                    pre_windows.append(windows+1)
                    Threshold = t_old+ (t_new - t_old)*f(0, pre_windows,alpha)
                    t_store = Threshold
                    t_old = sum(previous_threshold)/len(previous_threshold)
                    start_time = time.time()
                    windows = 0
                    previous_threshold = [Threshold]
        else:
            if(tot_str<=0):
                    tot_str = 0
                    windows  = 0
                    start_time = time.time()
                    stress.append(tot_str)
                    iter2.append(i)
                    previous_threshold = [t_store]
            if(tot_str>0 and tot_str<Threshold):
                    windows = windows+1 
                    Threshold = t_old+ (t_new - t_old)*f(windows, pre_windows, alpha)
                    previous_threshold.append(Threshold)
            if(tot_str>Threshold):    
                    end_time = time.time()
                    t_new = end_time- start_time
                    print("Have a change")
                    tot_str=0 
                    pre_windows.append(windows+1)
                    Threshold = t_old+ (t_new - t_old)*f(0, pre_windows, alpha)
                    t_old = sum(previous_threshold)/len(previous_threshold)
                    t_store = Threshold
                    windows = 0
                    previous_threshold = [Threshold]
        print(i, windows, tot_str, Threshold)
        stress1 = 0
        #time.sleep(5)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()