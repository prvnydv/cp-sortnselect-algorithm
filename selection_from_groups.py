import os
from image_frequency import img_frequency
from operator import or_
from functools import reduce
import shutil
from happy import expression_image
import pandas as pd
from face_extraction import img_to_faces
import numpy as np
from utils import unpack


def single_select(image_dir):
	# Selecting single image from groups with 3 or less number of images
    face_dir= 'face_image'
    os.makedirs(face_dir)
    path= image_dir+'/'
    files = os.listdir(path)
    for file in files:
        _,ext=file.split(".")
        if ext in ['jpg','JPG','jpeg','JPEG']:
            img_to_faces(path,file)

    # Selecting the best image based on emotion score
    index=[]
    image_id=[]
    happy=[]
    face_path=face_dir+'/'
    files=os.listdir(face_path)
    for file in files:
        name=[]
        name=file.split("$")
        happy.append(expression_image(face_path+file)[0])
        index.append(name[0])
        image_id.append(name[1])

    data= list(zip(index,image_id,happy))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
    grouped = final_data.groupby('image_id')
    happy=pd.DataFrame(grouped['happy'].agg(np.mean))
    happy=happy.sort_values(by=['happy'], ascending=False)
    shutil.rmtree(face_dir)
    return happy.index[0]


def happy_selection(image_dir,files):
    face_dir= 'face_image'
    os.makedirs(face_dir)
    path= image_dir+'/'
    for file in files:
        _,ext=file.split(".")
        if ext in ['jpg','JPG','jpeg','JPEG']:
            img_to_faces(path,file)

    index=[]
    image_id=[]
    happy=[]
    face_path=face_dir+'/'
    files=os.listdir(face_path)
    for file in files:
        name=[]
        name=file.split("$")
        happy.append(expression_image(face_path+file)[0])
        index.append(name[0])
        image_id.append(name[1])

    data= list(zip(index,image_id,happy))
    final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
    grouped = final_data.groupby('image_id')
    happy=pd.DataFrame(grouped['happy'].agg(np.mean))
    shutil.rmtree(face_dir)
    return happy


def selection_from_groups(image_dir):
    imgs=os.listdir(image_dir)
    face_dir='face_image'
    # If 1 image group then we select it 
    if len(imgs)==1:
        return imgs[0]
    # If 3 or lrss image group then single image selection based on emotion scores
    elif len(imgs)<4:
        return single_select(image_dir)
    # If none of the above then multiple selection 
    # Images that have less occuring faces are selected above 0.75 emotion score threshold
    else:
        x=[]
        a=dict()
        os.makedirs(face_dir)
        path= image_dir+'/'
        files = os.listdir(path)
        for i in range(len(files)):
        	# extracting all faces with their image maps i.e images in which that face is occuring 
            _,ext=files[i].split(".")
            if ext in ['jpg','JPG','jpeg','JPEG']:
                img_frequency(path,files[i],a,i)
        a_list=list(a.keys())
        for i in range(len(a_list)):
            a[a_list[i]]['images']=set(a[a_list[i]]['images'])

        x=[]
        for i in range(len(a_list)):
        	# selecting only faces that occur in less than 50% of the total number of images in that group
            if len(a[a_list[i]]['images'])< 0.5*len(os.listdir(image_dir)):
                x.append(a[a_list[i]]['images'])
        shutil.rmtree(face_dir)
        h=[]
        if len(x)>0:
            h=list(reduce(or_, x))
        # If no face occurs less than 50% of the total number of images in that group then we go back to single selection
        if len(h)==0:
            return single_select(image_dir)
        else:
        # Else we select all images above 0.75 threshold emotion score
            a=happy_selection(image_dir,h)
            a.reset_index(inplace=True)
            for i in range(len(a)):
                if a.iloc[i]['happy']> 0.75:
                    x.append(a.iloc[i]['image_id'])
            return list(set(unpack(x)))