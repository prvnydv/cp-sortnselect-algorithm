import face_recognition 
import os
import cv2
from face_extraction import detect_blur_fft
import numpy as np

model_face_extraction = cv2.dnn.readNetFromCaffe("face_extraction_model/deploy.prototxt", 'face_extraction_model/weights.caffemodel')
def img_frequency(path,file,a,j):
    image = cv2.imread(path + file)
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
                cv2.imwrite('face_image'+'/' + str(i) + '$' + file, frame)
                try:
                    # Saving New Faces and appending in image maps when similar image is found
                    other_img=face_recognition.load_image_file('face_image'+'/' + str(i) + '$' + file)
                    other_enc=face_recognition.face_encodings(other_img)[0]
                    a_list=list(a.keys())
                    flag=0
                    for k in range(len(a_list)):
                        if face_recognition.compare_faces([a[a_list[k]]['face_vector']], other_enc)[0]:
                            # Append in Image map if faces are similar
                            [a[a_list[k]]['images']][0].append(file)
                            flag=1
                            break
                        else:
                            continue
                    if flag==0:
                        # Save new face
                        a[str(i) + '$' + file]={'face_vector':other_enc,'images':[file]}

                                                
                    
                except:
                    pass