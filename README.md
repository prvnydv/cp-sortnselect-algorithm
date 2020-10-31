# cp-sortnselect-algorithm
A data science application for sorting and selection of relevant photos

## Ideation 
- Remove All files that are not images to avoid exceptions.
- Remove All images in which a face is not detected.
- Remove near duplicate images.
- Sort Images Based on Timestamp.
- Grouping the sorted images based on the top 10 colors in the images.
- Selecting Images Based on Emotion Score.
- Sorting the Selected Images Based on Features Extracted from the images such as Emotion, Overall Gender, Age & if images are candid or not.
- Creating a face count from the selected images to make an image cloud that can be presented to the client.

## Resources and Models
- Duplicate Removal 
  - Extracting Image Hashes from [Imagehash](https://pypi.org/project/ImageHash/) & clustering them based on the hash distances.
- Emotion Score
  - This score indicates if the subject is happy --> 1 or not --> 0.
  - The model used for this is trained on the [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)
- Gender
  - This score indicates if the subject is Female --> 1 or Male --> 0.
  - The model used for this is trained on the [UTKFace Dataset](https://www.kaggle.com/jangedoo/utkface-new)
- Age
  - This score indicates the age group of the subject in ascending order.
  - The model used in the purpose is a pretrained [OpenCV CNN](https://gist.github.com/GilLevi/c9e99062283c719c03de)
- Candid
  - This score indicates how much the eyes of the subject is visible in the image indicating candid-ness.
  - This is achieved by using [dlib landmark detector](http://dlib.net/face_landmark_detection.py.html) to calculate the EAR (Eyes Aspect Ratio).
 ## Execution
 The Execution of the whole pipeline is very simple.<br>
 `python main.py -i [image_folder] -n [number_of_images_in_output_required]`<br>
 Here only two arguements are needed, -i is followed by the image folder name and -n indicates the number of images needed as output.<br>
 **Note: Both the arguments are required and '[]' are for demonstration purposes here and not needed during execution**
