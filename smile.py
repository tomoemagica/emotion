import sys
import os
from os import path
from pathlib import Path, PureWindowsPath
from shutil import move
import cv2

target_dir = os.getcwd()
target_dir = os.path.join(target_dir, 'data_src', 'aligned')

file_count = len(os.listdir(target_dir))

print("Checking " + str(file_count) + " files")

smile_path = os.path.join(target_dir, 'smile')
neutral_path = os.path.join(target_dir, 'neutral')
cross_path = os.path.join(target_dir, 'cross')

if not path.isdir(smile_path):
    try:
        os.mkdir(smile_path)
    except OSError:
        print("Creation of the directory %s failed" % smile_path)
    else:
        print("Successfully created the directory %s " % smile_path)

if not path.isdir(neutral_path):
    try:
        os.mkdir(neutral_path)
    except OSError:
        print("Creation of the directory %s failed" % neutral_path)
    else:
        print("Successfully created the directory %s " % neutral_path)

if not path.isdir(cross_path):
    try:
        os.mkdir(cross_path)
    except OSError:
        print("Creation of the directory %s failed" % cross_path)
    else:
        print("Successfully created the directory %s " % cross_path)


k = [0, 0, 0]  # Blank
r = [255, 0, 0]  # Red
y = [255, 127, 0]  # Yellow
g = [0, 255, 0]  # Green

cross = [
    k, r, k, k, k, k, r, k,
    r, r, r, k, k, r, r, r,
    k, r, r, r, r, r, r, k,
    k, k, r, r, r, r, k, k,
    k, k, r, r, r, r, k, k,
    k, r, r, r, r, r, r, k,
    r, r, r, k, k, r, r, r,
    k, r, k, k, k, k, r, k
]

neutral_face = [
    k, k, y, y, y, y, k, k,
    k, y, k, k, k, k, y, k,
    y, k, y, k, k, y, k, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, k, y, y, y, y, k, y,
    k, y, k, k, k, k, y, k,
    k, k, y, y, y, y, k, k
]

smile_face = [
    k, k, g, g, g, g, k, k,
    k, g, k, k, k, k, g, k,
    g, k, g, k, k, g, k, g,
    g, k, k, k, k, k, k, g,
    g, k, g, k, k, g, k, g,
    g, k, k, g, g, k, k, g,
    k, g, k, k, k, k, g, k,
    k, k, g, g, g, g, k, k
]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

for thisFile in os.listdir(target_dir):
    file_name = os.path.join(target_dir, thisFile)
    if os.path.isfile(file_name):
        file_name = os.path.join(target_dir, thisFile)

        img_color = cv2.imread(file_name)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(45, 45))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                faceimg_color = img_color[y:y + h, x:x + w]
                faceimg_gray = img_gray[y:y + h, x:x + w]

                smiles = smile_cascade.detectMultiScale(
                    faceimg_gray, scaleFactor=1.7, minNeighbors=3, minSize=(15, 15))

                if len(smiles) > 0:
                    smile_file = os.path.join(smile_path, thisFile)
                    if os.path.isfile(file_name):
                        move(
                            file_name, smile_file)
                else:
                    neutral_file = os.path.join(neutral_path, thisFile)
                    if os.path.isfile(file_name):
                        move(
                            file_name, neutral_file)
        elif os.path.isfile(file_name):
            cross_file = os.path.join(cross_path, thisFile)
            move(
                file_name, cross_file)
