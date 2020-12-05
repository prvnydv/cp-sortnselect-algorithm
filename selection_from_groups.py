import os
from image_frequency import img_frequency
from happy import expression_image
import pandas as pd
from face_extraction import img_to_faces
import numpy as np
from utils import unpack
from utils import s3_client


def single_select(individual_group_folder, job_uid):
	# Selecting single image from groups with 3 or less number of images
    files = s3_client.ls(individual_group_folder)
    group_number = individual_group_folder.split("/")[-1]
    for file in files:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if file.endswith(ext):
            img_to_faces(job_uid, s3_uri, f"single_selection_from_groups/{group_number}")

    # Selecting the best image based on emotion score
    index=[]
    image_id=[]
    happy=[]
    face_path= f"s3://sns-outputs/{job_uid}/single_selection_from_groups/{group_number}"
    faces=s3_client.ls(face_path)
    for face in faces:
        name=face.split("/")[-1].split("$")
        happy.append(expression_image(face)[0])
        index.append(name[0])
        image_id.append(name[1])

    data= list(zip(index,image_id,happy))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
    grouped = final_data.groupby('image_id').agg({"happy" : np.mean}).reset_index()
    happy=grouped.sort_values(by=['happy'], ascending=False)
    return happy.iloc[0]['image_id']


def happy_selection(individual_group_folder, job_uid, sub_select_image_ids):
    files = s3_client.ls(individual_group_folder)
    group_number = individual_group_folder.split("/")[-1]
    for file in files:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if file.endswith(ext):
            img_to_faces(job_uid, s3_uri, f"happy_selection_from_groups/{group_number}")

    index=[]
    image_id=[]
    happy=[]
    face_path= f"s3://sns-outputs/{job_uid}/happy_selection_from_groups/{group_number}"
    files=s3_client.ls(face_path)
    for face in faces:
        name=[]
        name=face.split("/")[-1].split("$")
        if name[1] in sub_select_image_ids:
            happy.append(expression_image(face)[0])
            index.append(name[0])
            image_id.append(name[1])

    data= list(zip(index,image_id,happy))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
    grouped = final_data.groupby('image_id')
    happy=pd.DataFrame(grouped['happy'].agg(np.mean))

    return happy


def selection_from_groups(individual_group_folder, job_uid):
    imgs=s3_client.ls(individual_group_folder)
    # If 1 image group then we select it 
    if len(imgs)==1:
        return imgs[0].split("/")[-1]
    # If 3 or lrss image group then single image selection based on emotion scores
    elif len(imgs)<4:
        return single_select(individual_group_folder, job_uid)
    # If none of the above then multiple selection 
    # Images that have less occuring faces are selected above 0.75 emotion score threshold
    else:
        face_image_mapper=dict()
        for i in range(len(imgs)):
        	ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
            if file.endswith(ext):
                face_image_mapper = img_frequency(imgs[i], job_uid, face_image_mapper)

        face_image_list=list(face_image_mapper.keys())
        for i in range(len(face_image_list)):
            face_image_mapper[face_image_list[i]]['images']=set(face_image_mapper[face_image_list[i]]['images'])

        x=[]
        for i in range(len(face_image_list)):
        	# selecting only faces that occur in less than 50% of the total number of images in that group
            if len(face_image_mapper[face_image_list[i]]['images'])< 0.5*len(imgs)):
                x.append(face_image_mapper[face_image_list[i]]['images'])

        h=[]
        if len(x)>0:
            sub_select_image_ids=list(set(x))
        # If no face occurs less than 50% of the total number of images in that group then we go back to single selection
        if len(h)==0:
            return single_select(individual_group_folder, job_uid)
        else:
        # Else we select all images above 0.75 threshold emotion score
            a=happy_selection(individual_group_folder, job_uid, sub_select_image_ids)
            a.reset_index(inplace=True)
            for i in range(len(a)):
                if a.iloc[i]['happy']> 0.75:
                    x.append(a.iloc[i]['image_id'])
            return list(set(unpack(x)))