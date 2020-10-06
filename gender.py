from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np


model_gender = load_model('feature_models/gender.h5')
def gender_pred(image_directory):
    test_image = load_img(image_directory, target_size = (200,200))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model_gender.predict(x = test_image)
    return result