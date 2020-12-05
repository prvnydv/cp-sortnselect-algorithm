import dlib
from scipy.spatial import distance
import os
from cv2 import cv2
from imutils import paths,resize
from utils import read_with_cv2_from_generated_temp_file, write_cv2_image_to_s3

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("feature_models/shape_predictor_68_face_landmarks.dat")

def eyes_dir(s3_uri):
    image = read_with_cv2_from_generated_temp_file(s3_uri)
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    if len(faces)!=0:
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []

            for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(image,(x,y),(x2,y2),(0,255,0),1)

            for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(image,(x,y),(x2,y2),(0,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)
            return EAR
    return 0