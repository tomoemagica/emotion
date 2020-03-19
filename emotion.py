import numpy as np
import cv2
from keras.preprocessing import image
import dlib
from imutils import face_utils
import imutils
from sklearn import preprocessing
import math
from keras.models import model_from_json
import os
import sys
from os import path
import sys
from pathlib import Path, PureWindowsPath
from shutil import move
import tensorflow as tf
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.logging.set_verbosity(tf.logging.FATAL)
tf.get_logger().setLevel(logging.ERROR)

#-----------------------------
#opencv initialization
pythonpath = "C:/Program Files/Python-3.7.6/"
progroot = os.path.join(pythonpath, 'Facial-Expression-Keras')

face_cascade = cv2.CascadeClassifier(os.path.join(progroot, 'haarcascade_frontalface_default.xml'))

#-----------------------------
#face expression recognizer initialization
# Using pretrained model
model_path = os.path.join(progroot, 'model')
model = model_from_json(open(os.path.join(model_path, 'model.json'), "r").read())
model.load_weights(os.path.join(model_path, 'model.h5'))  #load weights

#-----------------------------

emotions = ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
# initialize dlib's face detector and create a predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(progroot + '/shape_predictor_68_face_landmarks.dat')


def detect_parts(image):
    distances = []
    # resize the image, and convert it to grayscale
    image = imutils.resize(image, width=200, height=200)
	
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        distances = euclidean_all(shape)
        # visualize all facial landmarks with a transparent overlay
        #output = face_utils.visualize_facial_landmarks(image, shape)
        #cv2.imshow("Image", output)
        #cv2.waitKey(0)	
        return distances

def euclidean(a, b):
   dist = math.sqrt(math.pow((b[0] - a[0]), 2) + math.pow((b[1] - a[1]), 2))
   return dist 

# calculates distances between all 68 elements
def euclidean_all(a):  
    distances = ""
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            dist = euclidean(a[i], a[j])
            dist = "%.2f" % dist;
            distances = distances + " " + str(dist)
    return distances

target_dir = os.getcwd()
target_dir = os.path.join(target_dir, 'data_src')
target_dir = os.path.join(target_dir, 'aligned')

Angry_path = os.path.join(target_dir, 'Angry')
Disgust_path = os.path.join(target_dir, 'Disgust')
Fear_path = os.path.join(target_dir, 'Fear')
Happy_path = os.path.join(target_dir, 'Happy')
Neutral_path = os.path.join(target_dir, 'Neutral')
Sad_path = os.path.join(target_dir, 'Sad')
Surprise_path = os.path.join(target_dir, 'Surprise')

if not path.isdir(Angry_path):
   try:
       os.mkdir(Angry_path)
   except OSError:
       print("Creation of the directory %s failed" % Angry_path)
   else:
       print("Successfully created the directory %s " % Angry_path)

if not path.isdir(Disgust_path):
   try:
       os.mkdir(Disgust_path)
   except OSError:
       print("Creation of the directory %s failed" % Disgust_path)
   else:
       print("Successfully created the directory %s " % Disgust_path)

if not path.isdir(Fear_path):
   try:
       os.mkdir(Fear_path)
   except OSError:
       print("Creation of the directory %s failed" % Fear_path)
   else:
       print("Successfully created the directory %s " % Fear_path)

if not path.isdir(Happy_path):
   try:
       os.mkdir(Happy_path)
   except OSError:
       print("Creation of the directory %s failed" % Happy_path)
   else:
       print("Successfully created the directory %s " % Happy_path)

if not path.isdir(Neutral_path):
   try:
       os.mkdir(Neutral_path)
   except OSError:
       print("Creation of the directory %s failed" % Neutral_path)
   else:
       print("Successfully created the directory %s " % Neutral_path)

if not path.isdir(Sad_path):
   try:
       os.mkdir(Sad_path)
   except OSError:
       print("Creation of the directory %s failed" % Sad_path)
   else:
       print("Successfully created the directory %s " % Sad_path)

if not path.isdir(Surprise_path):
   try:
       os.mkdir(Surprise_path)
   except OSError:
       print("Creation of the directory %s failed" % Surprise_path)
   else:
       print("Successfully created the directory %s " % Surprise_path)

file_count = len(os.listdir(target_dir))

print("Checking " + str(file_count) + " files")


for thisFile in os.listdir(target_dir):
    file_name = os.path.join(target_dir, thisFile)
    if os.path.isfile(file_name):
        file_name = os.path.join(target_dir, thisFile)

        img = cv2.imread(file_name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        distances = detect_parts(detected_face)

        if not distances is None:
            if(len(distances) != 0):
                val = distances.split(" ")[1:]
                val = np.array(val)
                val = val.astype(np.float)
                val = np.expand_dims(val, axis = 1)			
                minmax = preprocessing.MinMaxScaler()
                val = minmax.fit_transform(val)
                val = val.reshape(1,4624)

                predictions = model.predict(val) #store probabilities of 6 expressions
        		#find max indexed array ( 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise')
                max_index = np.argmax(predictions[0])
                emotion = emotions[max_index]

                if os.path.isfile(file_name):
                    if emotion == 'Happy':
                        move(
                            file_name, Happy_path)
                    elif emotion == 'Neutral':
                        move(
                            file_name, Neutral_path)
                    elif emotion == 'Sad':
                        move(
                            file_name, Sad_path)
                    elif emotion == 'Fear':
                        move(
                            file_name, Fear_path)
                    elif emotion == 'Angry':
                        move(
                            file_name, Angry_path)
                    elif emotion == 'Disgust':
                        move(
                            file_name, Disgust_path)
                    elif emotion == 'Surprised':
                        move(
                            file_name, Surprised_path)
