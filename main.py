
from flask import *
from datetime import datetime
from flask import copy_current_request_context
from dublicate import *
from face_extraction import img_to_faces
from utils import get_date_taken,unpack
from selection_from_groups import selection_from_groups
from final_selection import selection
from image_frequency import img_frequency
import time
import json
from similarity import get_colors,color_diff
import cv2
from utils import s3_client
from utils import read_with_cv2_from_generated_temp_file, write_cv2_image_to_s3
from utils import list_all_objects_of_a_bucket_folder, list_key_bucket_object
import requests
from threading import Thread

app = Flask(__name__)



@app.route("/sns/v1", methods = ['POST'])
def return_status():
    @copy_current_request_context        
    def consolidated_score():
        url_array = request.json['url_array']
        number_of_output_images = request.json['number_of_output_images']
        job_uid = request.json['job_uid']
        process_id = request.json['process_id']
        date_format = "%Y-%m-%d %H:%M:%S"
        current_time = datetime.strftime(datetime.now(), date_format)
        

        start_time = time.time()

        print(f" Input Image urls {url_array}")
        print(f" number_of_output_images :: {number_of_output_images}")
        print(f" Job UID :: {job_uid}")
        url_image_id_mapper = {}
        for url in url_array:
            image_id = url.split("/")[-1]
            url_image_id_mapper[image_id] = url

        print("Generated Image Url mapper")
        ##################################################################### Removing All files except images ###################################################################################################################################################
        for url in url_array:
            ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
            if not url.endswith(ext):
                url_array.remove(url)

        print(f"size of image_url is {len(url_array)}")

        ############################################################################ Dublication Removal ###################################################################################################################################################


        for i in range(2):
            # 16 --> Hash Length
            # 240 --> Threshold
            url_array = remove_similar_from_dir(url_array,16,240,job_uid)
        print(f"size of w/o_dup_image_url is {len(url_array)}")

        ##################################################################### Removing All Images except images that have faces in them ###################################################################################################################################################

        for url in url_array:
            ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
            if url.endswith(ext):
                print(f'Running Face Extraction Model')
                img_to_faces(job_uid, url, 'image_faces')   



        image_id=[]
        face_files_url=list_all_objects_of_a_bucket_folder('pical-backend-dev', 'image_faces')
        for url in face_files_url:
            name=url.split("$")
            image_id.append(name[-1])
        image_id=set(image_id)

        only_filenames = [url.split("/")[-1] for url in url_array]
        images_without_people = []
        for i in range(len(only_filenames)):
            if only_filenames[i] not in image_id:
                url_element = f's3://pical-backend-dev/store/{only_filenames[i]}'
                images_without_people.append(url_element)
                url_array.remove(url_element)  

        ##################################################################### Sorting Images based on timestamp ###################################################################################################################################################

        date=[]
        image_id=[]
        for url in url_array:
            try:
                date.append(get_date_taken(url))
            except:
                pass
            image_id.append(url.split("/")[-1])
        if len(date) > 0:
            all_features= list(zip(image_id,date))
            all_features=pd.DataFrame(all_features)
            all_features.columns=["image_id","date_time"]
            all_features = all_features.sort_values(by=['date_time'], ascending=True)
        else:
            all_features= list(zip(image_id))
            all_features=pd.DataFrame(all_features)
            all_features.columns=["image_id"]
        # #################################################################### Grouping Images based on Color Palette  ###################################################################################################################################################

        grouped_url_array = {}
        group_number=1
        group=f"group_{group_number}"
        grouped_url_array[group] = []
        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[0]['image_id']])
        if len(all_features) > 3:
            for i in range(len(all_features)-3):
                a,b,c,d=get_colors(url_image_id_mapper[all_features.iloc[i]['image_id']]), get_colors(url_image_id_mapper[all_features.iloc[i+1]['image_id']]), get_colors(url_image_id_mapper[all_features.iloc[i+2]['image_id']]), get_colors(url_image_id_mapper[all_features.iloc[i+3]['image_id']])
                # Checking ith and (i+3)th image
                if color_diff(a,d)>5: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+1]['image_id']])
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+2]['image_id']])
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+3]['image_id']])

                        i+=2
                    except:
                        pass
                # Checking ith and (i+2)th image
                elif color_diff(a,c)>5: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+1]['image_id']])
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+2]['image_id']])

                        i+=1
                    except:
                        pass
                # Checking ith and (i+1)th image
                elif color_diff(a,b)>5: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+1]['image_id']])
                    except:
                        pass 
                else: # New group created if none of the above criterion are met
                    group_number+=1
                    try:
                        grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[i+1]['image_id']])
                    except:
                        pass
        else:
            for index,element in enumerate(all_features['image_id'].tolist()):
                group=f"group_{group_number+index+1}"
                grouped_url_array[group] = []
                grouped_url_array[group].append(url_image_id_mapper[all_features.iloc[index+1]['image_id']])

        # #################################################################### Selection of Images from the groups ###################################################################################################################################################
        images = []
        print(f"Length of all groups :: {len(list(grouped_url_array.keys()))}")
        for key, val in grouped_url_array.items():
            print(f"Total images in the {key} :: {len(val)}")
            images.append(selection_from_groups(val))
        images=list(unpack(images)) 

        # #################################################################### Sorting The Images ###################################################################################################################################################
        print("################################################################################################################")
        print(f"Final length of selected images :::::::: {len(images)}")
        print("################################################################################################################")
        
        image_urls = [url_image_id_mapper[id] for id in images]
        final_selection=selection(image_urls, job_uid)   
        print("################################################################################################################")
        print(f"Final length of sorted df :::::::: {len(final_selection)}")
        print("################################################################################################################")

        final_selection = final_selection[0:int(number_of_output_images)]


        # ############################################################ Creating the output  ###################################################################################################################################################

        final_selection['image_s3_url'] = final_selection.apply(lambda row: url_image_id_mapper[row['image_id']], axis=1)

        result = {"image_processing_selection_sorting_process" : {
            "output_raw_json" : str({
                'number_of_output_images' : number_of_output_images,
                'output_image_urls' : final_selection['image_s3_url'].tolist(),
                'images_without_people' : images_without_people,
                'sns version' : 'v1.0',
                'timestamp' : current_time }),
            "status" : "success"
        }}

        print("hello world")
        url = f"http://localhost:5000/image_processing/selection_sorting_processes/{process_id}"
        x = requests.patch(url, json=result)

    """Return first the response and tie the consolidated_score to a thread"""
    Thread(target = consolidated_score).start()
    return jsonify({"status" : "ongoing", "message" : "We have Received the request. Please wait for 5 minutes wfor Job to complete"})
    
if __name__ == '__main__':
    app.run(debug=True, port=8877, host = '0.0.0.0')

