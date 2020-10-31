from imutils import paths,resize
import cv2
import os
import numpy as np

def detect_blur_fft(image, size=60, thresh=0, vis=True):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean

model_face_extraction = cv2.dnn.readNetFromCaffe("face_extraction_model/deploy.prototxt", 'face_extraction_model/weights.caffemodel')
def img_to_faces(path,file):
	try:
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
	            if len(frame)>200 and len(frame[0])>200 and mean>0: # Face dimentions should be atleast (200,200) and blur> 0
	                cv2.imwrite('face_image'+'/' + str(i) + '$' + file, frame)
	except:
		pass