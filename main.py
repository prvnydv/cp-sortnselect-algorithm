
from flask import request
from flask import Flask, jsonify
from datetime import datetime

from dublicate import *
from face_extraction import img_to_faces
from utils import get_date_taken,unpack
from selection_from_groups import selection_from_groups
from final_selection import selection
from image_frequency import img_frequency
import time
from similarity import get_colors,color_diff
import cv2
import shutil
from utils import s3_client
from utils import read_with_cv2_from_generated_temp_file, write_cv2_image_to_s3

app = Flask(__name__)



@app.route("/sns/v1", methods = ['POST'])
def consolidated_score():
    url_array = request.json['url_array']
    number_of_output_images = request.json['number_of_output_images']
    job_uid = request.json['job_uid']
    date_format = "%Y-%m-%d %H:%M:%S"
    current_time = datetime.strftime(datetime.now(), date_format)
    

    start_time = time.time()

    url_image_id_mapper = {}
    for url in url_array:
        image_id = url.split("/")[-1]
        url_image_id_mapper[image_id] = url

    ##################################################################### Removing All files except images ###################################################################################################################################################

    for url in url_array:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if not url.endswith(ext):
            url_array.remove(url)



    ############################################################################ Dublication Removal ###################################################################################################################################################


    for i in range(2):
        # 16 --> Hash Length
        # 240 --> Threshold
        duplicate_images = remove_similar_from_dir(url_array,16,240,job_uid)
        url_array.remove(duplicate_images)


    ##################################################################### Removing All Images except images that have faces in them ###################################################################################################################################################

    for url in url_array:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if url.endswith(ext):
            img_to_faces(job_uid, url, 'image_with_faces')   



    image_id=[]
    face_files_url=s3_client.ls(f"s3://sns-outputs/{job_uid}/faces_extracted")
    for url in face_files_url:
        name=url.split("$")
        image_id.append(name[1].split(".")[0])
    image_id=set(image_id)

    only_filenames = [url.split("/")[-1].split(".")[0] for url in url_array]
    for i in range(len(only_filenames)):
        if only_filenames[i] not in image_id:
            del url_array[i]  

    ##################################################################### Sorting Images based on timestamp ###################################################################################################################################################

    date=[]
    image_id=[]
    for url in url_array:
        date.append(get_date_taken(url))
        image_id.append(url.split("/")[-1])
    all_features= list(zip(image_id,date))
    all_features=pd.DataFrame(all_features)
    all_features.columns=["image_id","date_time"]
    all_features = all_features.sort_values(by=['date_time'], ascending=True)

    # #################################################################### Grouping Images based on Color Palette  ###################################################################################################################################################


    group_number=1
    group='group_test'
    img = read_with_cv2_from_generated_temp_file(all_features.iloc[0]['image_id'])
    write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[0]['image_id']}", job_uid)
    for i in range(len(all_features)-3):
        a,b,c,d=get_colors(all_features.iloc[i]['image_id']),get_colors(all_features.iloc[i]['image_id']),get_colors(all_features.iloc[i]['image_id']),get_colors(all_features.iloc[i]['image_id'])
        # Checking ith and (i+3)th image
        if color_diff(a,d)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+1]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+1]['image_id']}", job_uid)
                
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+2]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+2]['image_id']}", job_uid)
                
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+3]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+3]['image_id']}", job_uid)
                i+=2
            except:
                pass
        # Checking ith and (i+2)th image
        elif color_diff(a,c)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+1]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+1]['image_id']}", job_uid)
                
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+2]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+2]['image_id']}", job_uid)

                i+=1
            except:
                pass
        # Checking ith and (i+1)th image
        elif color_diff(a,b)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+1]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+1]['image_id']}", job_uid)
            except:
                pass 
        else: # New group created if none of the above criterion are met
            group_number+=1
            try:
                img = read_with_cv2_from_generated_temp_file(all_features.iloc[i+1]['image_id'])
                write_cv2_image_to_s3(img, group, f"{group_number}/{all_features.iloc[i+1]['image_id']}", job_uid)
            except:
                pass

    # #################################################################### Selection of Images from the groups ###################################################################################################################################################

    images=[]
    var='group_test'
    job_group_folder = f"s3://sns-outputs/{job_uid}/{var}"
    groups = s3_client.ls(job_group_folder)
    for i in range(len(groups)):
        individual_group_folder = f"{job_group_folder}/{i}"
        images.append(selection_from_groups(individual_group_folder, job_uid))
    images=list(unpack(images)) 

    # #################################################################### Sorting The Images ###################################################################################################################################################

    image_urls = [url_image_id_mapper[id] for id in images]
    final_selection=selection(image_urls, job_uid)[int(number_of_output_images)]   


    # ############################################################ Creating the output  ###################################################################################################################################################

    final_selection['image_s3_url'] = final_selection.apply(lambda row: url_image_id_mapper[row['image_id']], axis=1)


    # #################################################################### Creating the Face Count for Image Cloud ###################################################################################################################################################
    # a=dict()
    # os.makedirs(face_dir)
    # final_selection=os.listdir(new_img_dir)
    # path= image_dir+'/'
    # for i in range(len(final_selection)):
    #     img_frequency(path,final_selection[i],a,i)
    # a_list=list(a.keys())
    # for i in range(len(a_list)):
    #     a[a_list[i]]['images']=set(a[a_list[i]]['images'])



    # print("--- %s seconds ---" % (time.time() - start_time))


    result = {
      "input_image_urls" : url_array,
      "number_of_input_images" : len(url_array),
      "number_of_output_images" : number_of_output_images,
      "output_image_urls" : final_selection['image_s3_url'].tolist(),
      "sns version" : "v1.0",
      "timestamp" : current_time
    }

    print("hello world")

    return jsonify(result=result)

    
if __name__ == '__main__':
    app.run(debug=True, port=8877, host = '0.0.0.0')

