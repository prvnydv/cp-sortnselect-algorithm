from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
from utils import read_pillow_image_from_s3


model_gender = load_model('feature_models/gender.h5')
def gender_pred(s3_uri):
    test_image = load_img(read_pillow_image_from_s3(s3_uri), target_size = (200,200))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model_gender.predict(x = test_image)
    return result