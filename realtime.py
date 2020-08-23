import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Model, load_model,Sequential, model_from_json
import cv2
import dlib
from PIL import Image
from skimage import transform
import json
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
#loading the model architecture and weights
json_file = open('model_feat4.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
json_file.close()
model.load_weights("model_feat4_weights.h5")
# model = load_model('model2\model_65_0.007784269750118256.h5')
def get_prediction(image,model):
    pred = model.predict(image)
    val = int(np.argmax(pred,axis=1))
    e = get_key(val)
    return e

def get_key(val): 
    d = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Sad', 4:'Surprise', 5: 'Neutral'}
    return d[val]
def realtime():
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    # count=0
    while True:
        # cap = cv2.VideoCapture(0)
        # net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        _, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # count+=1
        # print(count)
        # print(type(frame))
        (h,w) = frame.shape[:2]  
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (256,256)), 1.0, (256, 256), (104.0, 117.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence < 0.7:
                continue    
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX,startY,endX,endY)
            image = frame[startY:endY, startX:endX]
            print(image.shape)
            if image.shape is not  None :
                face = cv2.cvtColor(cv2.resize(image,(48,48)), cv2.COLOR_BGR2GRAY)
            roi = face.astype("float") / 255.0
            roi = np.reshape(roi,(1,48,48,1))
            # print(roi.shape)
            emotion = get_prediction(roi,model)
            # print(emotion)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),1)
            cv2.putText(frame, emotion, (startX,y),cv2.FONT_HERSHEY_COMPLEX,0.9,(255,0,0),2)
            # print(type(image))
        cv2.imshow("Frame" , frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()   
    cap.release()
    
if __name__=='__main__':
    realtime()