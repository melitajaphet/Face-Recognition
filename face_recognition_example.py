# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:15:16 2020

@author: Melita Japhet
"""

import face_recognition
import os
import cv2

known_faces_dir = "known_faces"
#unknown_faces_dir = "unknown_faces" #for image face-recognition
tolerance = 0.45
frame_thickness = 3
font_thickness = 2
MODEL = "cnn"

video = cv2.VideoCapture(0)

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f'{known_faces_dir}/{name}'):
        image = face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
        
print("processing known faces")
#for filename in os.listdir(unknown_faces_dir): #for image face-recognition
while True:
    #print(filename) #for image face-recognition
    #image = face_recognition.load_image_file(f'{unknown_faces_dir}/{filename}') #for image face-recognition
    
    ret, image = video.read()
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image,locations)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #for image face-recognition
    
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), font_thickness)
            
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    #cv2.waitKey(0) #for image face-recognition
    #cv2.destroyWindow(filename) #for image face-recognition
            