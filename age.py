import cv2
from utils import read_with_cv2_from_generated_temp_file


ageNet = cv2.dnn.readNet( 'feature_models/age_net.caffemodel',"feature_models/age_deploy.prototxt")
ageList=[1,1,2,2,3,4,5,6]
def face_age(s3_uri):
    image = read_with_cv2_from_generated_temp_file(s3_uri)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
    
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    return age