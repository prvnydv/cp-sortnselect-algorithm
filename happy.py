from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import time
from utils import load_image_for_keras


expression_model=load_model('feature_models/fer.h5')
def expression_image(s3_uri):
    test_image = load_image_for_keras(s3_uri=s3_uri, target_size=(48,48))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = (0,3))
    test_image=test_image[:,:,:,:,0]
    test_image=test_image/255
    result = expression_model.predict(x=test_image)[0]
    print(f'result of expression model :: {expression_model.predict(x=test_image)}')
    return (result)