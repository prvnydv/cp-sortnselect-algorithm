import os
from image_frequency import img_frequency
from happy import expression_image
import pandas as pd
from face_extraction import img_to_faces
import numpy as np
from utils import unpack
from utils import s3_client
from utils import list_all_objects_of_a_bucket_folder



def single_select(imgs, job_uid):
	# Selecting single image from groups with 3 or less number of images
  index=[]
  image_id=[]
  happy=[]
  faces=list_all_objects_of_a_bucket_folder('pical-ds-dev', f'{job_uid}/image_faces')
  for face in faces:
    if face.split("/")[-1].split("$")[-1] in imgs:
      name=face.split("/")[-1].split("$")
      happy.append(expression_image(face)[0])
      index.append(name[0])
      image_id.append(name[1])

  data= list(zip(index,image_id,happy))
  final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
  print(f"Single select dataframe {final_data}")
  grouped = final_data.groupby('image_id').agg({"happy" : np.mean}).reset_index()
  happy=grouped.sort_values(by=['happy'], ascending=False)
  return happy.iloc[0]['image_id']


def happy_selection(images, job_uid):
  file_ids = images

  index=[]
  image_id=[]
  happy=[]
  faces=list_all_objects_of_a_bucket_folder('pical-ds-dev', f'{job_uid}/image_faces')
  for face in faces:
    if face.split("/")[-1].split("$")[-1] in file_ids:
      name=[]
      name=face.split("/")[-1].split("$")
      happy.append(expression_image(face)[0])
      index.append(name[0])
      image_id.append(name[1])

  data= list(zip(index,image_id,happy))
  final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
  grouped = final_data.groupby('image_id')
  happy=pd.DataFrame(grouped['happy'].agg(np.mean))

  return happy


def selection_from_groups(images, drive, job_uid):
  # If 1 image group then we select it 
  if len(images)==1:
    return images[0].split("/")[-1]
  # If 3 or lrss image group then single image selection based on emotion scores
  elif len(images)<4:
    return single_select(images, job_uid)
  # If none of the above then multiple selection 
  # Images that have less occuring faces are selected above 0.75 emotion score threshold
  else:
    face_image_mapper=dict()
    for i in range(len(images)):
      face_image_mapper = img_frequency(images[i], drive, face_image_mapper)

    face_image_list=list(face_image_mapper.keys())
    for i in range(len(face_image_list)):
      face_image_mapper[face_image_list[i]]['images']=set(face_image_mapper[face_image_list[i]]['images'])

      x=[]
      for i in range(len(face_image_list)):
        # selecting only faces that occur in less than 50% of the total number of images in that group
        if len(face_image_mapper[face_image_list[i]]['images'])< 0.5*len(images):
          x.append(face_image_mapper[face_image_list[i]]['images'])
      
      x = [img for l in x for img in l]
      sub_select_image_ids=[]
      print(f"Images containing selected faces are {x}")
      if len(x)>0:
        sub_select_image_ids=list(set(x))
      # If no face occurs less than 50% of the total number of images in that group then we go back to single selection
      if len(sub_select_image_ids)==0:
        return single_select(images, job_uid)
      else:
      # Else we select all images above 0.75 threshold emotion score
        a=happy_selection(sub_select_image_ids, job_uid)
        a.reset_index(inplace=True)
        for i in range(len(a)):
          if a.iloc[i]['happy']> 0.75:
            x.append(a.iloc[i]['image_id'])
        return list(set(unpack(x)))