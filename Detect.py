import cv2
import numpy as np

import matplotlib.pyplot as plt

#importing dependencies related to image transformations

from PIL import Image

#importing dependencies related to data loading


import YOLO

def detect_face(img):
    face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)

    faces=face_clsfr.detectMultiScale(gray,1.3,3)
    print(f'Number of faces found = {len(faces)}')
    if len(faces) == 0:
        return None

    

    x,y,w,h = 0, 0, 0, 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    

    face_img=img[y:y+w,x:x+w]
    #plt.imshow(face_img)
    #cv2.imshow("face found          ",face_img)
    #cv2.waitKey()
    return face_img







 

def Test():
    img = cv2.imread("demo1.jpg")
    #half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    #cv2.imshow("Original image",half)

    detect = YOLO.humanDetect(img)
    cv2.imshow("human image",detect)
    cv2.waitKey()

    detect = detect_face(detect)
    cv2.imshow("face", detect)
    cv2.waitKey()

#Test()
    

