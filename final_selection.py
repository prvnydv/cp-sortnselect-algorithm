import os
from face_extraction import img_to_faces
from happy import expression_image
import pandas as pd
from eyes import eyes_dir
from utils import s3_client
from gender import gender_pred
from age import face_age
import numpy as np
from utils import list_all_objects_of_a_bucket_folder

# Final sorting of all images that are selected from groups
def selection(urls, job_uid):
    for url in urls:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if url.endswith(ext):
            img_to_faces(job_uid, url, 'face_extraction_for_sorting')

    # Emotion scores of images 
    index=[]
    image_id=[]
    happy=[]
    face_count=[]
    files=list_all_objects_of_a_bucket_folder('pical-backend-dev', 'face_extraction_for_sorting')
    for file in files:
        name=[]
        name=file.split("/")[-1].split("$")
        happy.append(expression_image(file)[0])
        index.append(name[0])
        image_id.append(name[1])
        face_count.append(1)

    data= list(zip(index,image_id,happy,face_count))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy','face_count'])
    grouped = final_data.groupby('image_id')
    happy=pd.DataFrame(grouped['happy'].agg(np.mean))
    face_count= pd.DataFrame(grouped['face_count'].agg(np.sum))

    # Candid scores of images
    eyes_focus=[]

    for file in files:
        eyes_focus.append(eyes_dir(file))

    data= list(zip(index,image_id,eyes_focus))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','not_candid'])
    grouped = final_data.groupby('image_id')
    eyes=grouped['not_candid'].agg(np.mean)
    eyes=pd.DataFrame(eyes)


    # Gender scores of images
    gender=[]
    for file in files:
        result=gender_pred(file)
        gender.append(result[0][0][0])


    data= list(zip(index,image_id,gender))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','gender'])
    grouped = final_data.groupby('image_id')
    gender=pd.DataFrame(grouped['gender'].agg(np.mean))

    # Age scores of images
    age=[]
    for file in files:
        result=face_age(file)
        age.append(result)

    data= list(zip(index,image_id,age))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','age'])
    grouped = final_data.groupby('image_id')
    age=pd.DataFrame(grouped['age'].agg(np.mean))

    # Combining All the features
    # Weights of each feature :   Gender--> 0.1 Age--> 0.2   Candid--> 0.4   Emotion--> 0.3
    all_features2=[happy,eyes,age,gender,face_count]
    all_features2=pd.concat(all_features2,axis=1)
    all_features2['final_feature']=all_features2.apply(lambda row: row.gender*0.1 + 0.2/row.age+ row.not_candid*0.4+ row.happy*0.3, axis=1)
    all_features2 = all_features2.sort_values(by=['final_feature'], ascending=False) # Sorting Based on final feature
    all_features2.reset_index(inplace = True)
    
    return all_features2


