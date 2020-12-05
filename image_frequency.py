import face_recognition 
import os
import cv2
from face_extraction import detect_blur_fft
import numpy as np
from utils import read_pillow_image_from_s3
from utils import write_cv2_image_to_s3, read_with_cv2_from_generated_temp_file

model_face_extraction = cv2.dnn.readNetFromCaffe("face_extraction_model/deploy.prototxt", 'face_extraction_model/weights.caffemodel')
def img_frequency(s3_uri,job_uid,face_image_mapper):

    image = read_with_cv2_from_generated_temp_file(s3_uri)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = detect_blur_fft(gray, size=60)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model_face_extraction.setInput(blob)
    detections = model_face_extraction.forward()
    
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]
        count=0
        

      # If confidence > 0.5, save it as a separate file
        if (confidence > 0.5):
            count += 1
            frame = image[startY:endY, startX:endX]
            if len(frame)>200 and len(frame[0])>200 and mean>0:

                #write_cv2_image_to_s3(frame, folder, f"{i}${s3_uri.split("/")[-1]}', job_uid)
                try:
                    # Saving New Faces and appending in image maps when similar image is found
                    #other_img=face_recognition.load_image_file(read_pillow_image_from_s3(f"s3://pical-backend-dev/faces_extracted/{job_uid}/{i}${s3_uri.split("/")[-1]}"))
                    face_enc_of_image=face_recognition.face_encodings(frame)[0]
                    face_image_list = list(face_image_mapper.keys())
                    flag=0
                    for k in range(len(face_image_list)):
                        if face_recognition.compare_faces([face_image_mapper[face_image_list[k]]['face_vector']], face_enc_of_image)[0]:
                            # Append in Image map if faces are similar
                            face_image_mapper[face_image_list[k]]['images'].append(f"{s3_uri.split('/')[-1]}")
                            flag=1
                            break
                        else:
                            continue
                    if flag==0:
                        # Save new face
                        face_image_mapper[f"{i}${s3_uri.split('/')[-1]}"]={'face_vector':face_enc_of_image,'images':[f"{s3_uri.split('/')[-1]}"]}                               
                    
                except:
                    pass
    return face_image_mapper