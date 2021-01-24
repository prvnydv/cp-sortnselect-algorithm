
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
from utils import list_all_objects_of_a_bucket_folder, list_key_bucket_object
import requests
from threading import Thread
from utils import create_gdrive_instance

app = Flask(__name__)



@app.route("/sns/v1", methods = ['POST'])
def return_status():
    @copy_current_request_context        
    def consolidated_score():
        date_format = "%Y-%m-%d %H:%M:%S"
        drive = create_gdrive_instance()
        folder_to_execute = drive.ListFile({"q": "mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()[0]

        folder_name = folder_to_execute['title']
        folder_id   = folder_to_execute['id']

        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

        image_file_list = []
        for file in file_list:
            if file['mimeType'] in ['image/png', 'image/jpeg']:
                image_file_list.append(file)

        url_array = []
        url_image_id_mapper = {}
        for file in image_file_list:
            url_array.append(file['id'])
            url_image_id_mapper[file['id']] = file['title']
        
        number_of_output_images = 5
        job_uid = 8894

        start_time = time.time()

        print(f" Input Image urls {url_array}")
        print(f" number_of_output_images :: {number_of_output_images}")
        print(f" Job UID :: {job_uid}")
        
        print("Generated Image Url mapper")

        ############################################################################ Dublication Removal ###################################################################################################################################################


        for i in range(2):
            # 16 --> Hash Length
            # 240 --> Threshold
            url_array = remove_similar_from_dir(url_array,16,240,job_uid, drive)
        print(f"size of w/o_dup_image_url is {len(url_array)}")

        ##################################################################### Removing All Images except images that have faces in them ###################################################################################################################################################

        for url in url_array:
            print(f"Check the input url {url}")
            img_to_faces(job_uid, url, drive, 'image_faces')   



        image_id=[]
        face_files_url=list_all_objects_of_a_bucket_folder('pical-backend-dev', f'{job_uid}/image_faces')
        for url in face_files_url:
            name=url.split("$")
            image_id.append(name[-1])
        image_id=set(image_id)
        files_to_check = url_array.copy()
        images_without_people = []
        for i in range(len(files_to_check)):
            if files_to_check[i] not in image_id:
                images_without_people.append(files_to_check[i])
                url_array.remove(files_to_check[i])  

        print(f"Number of images with faces {url_array}")
        print(f"Number of images without faces {images_without_people}")

        ##################################################################### Sorting Images based on timestamp ###################################################################################################################################################

        date=[]
        image_id=[]
        for url in url_array:
            try:
                print(f"Date of the file :: {get_date_taken(url, drive)}")
                date.append(get_date_taken(url, drive))
            except:
                pass
            image_id.append(url)
        if len(date) > 0:
            all_features= list(zip(image_id,date))
            all_features=pd.DataFrame(all_features)
            all_features.columns=["image_id","date_time"]
            all_features = all_features.sort_values(by=['date_time'], ascending=True)
        else:
            all_features= list(zip(image_id))
            all_features=pd.DataFrame(all_features)
            all_features.columns=["image_id"]
        print(f"Input dataframe :: {all_features}")
        # #################################################################### Grouping Images based on Color Palette  ###################################################################################################################################################

        grouped_url_array = {}
        group_number=1
        group=f"group_{group_number}"
        grouped_url_array[group] = []
        grouped_url_array[group].append(all_features.iloc[0]['image_id'])
        if len(all_features) > 3:
            for i in range(len(all_features)-3):
                a,b,c,d=get_colors(drive, all_features.iloc[i]['image_id']), get_colors(drive, all_features.iloc[i+1]['image_id']), get_colors(drive, all_features.iloc[i+2]['image_id']), get_colors(drive, all_features.iloc[i+3]['image_id'])
                # Checking ith and (i+3)th image
                if color_diff(a,d)>18: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(all_features.iloc[i+1]['image_id'])
                        grouped_url_array[group].append(all_features.iloc[i+2]['image_id'])
                        grouped_url_array[group].append(all_features.iloc[i+3]['image_id'])

                        i+=2
                    except:
                        pass
                # Checking ith and (i+2)th image
                elif color_diff(a,c)>18: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(all_features.iloc[i+1]['image_id'])
                        grouped_url_array[group].append(all_features.iloc[i+2]['image_id'])

                        i+=1
                    except:
                        pass
                # Checking ith and (i+1)th image
                elif color_diff(a,b)>18: # 6 out of 10 colors should be same if they are to be in same group
                    try:
                        grouped_url_array[group].append(all_features.iloc[i+1]['image_id'])
                    except:
                        pass 
                else: # New group created if none of the above criterion are met
                    group_number+=1
                    try:
                        grouped_url_array[group].append(all_features.iloc[i+1]['image_id'])
                    except:
                        pass
        else:
            for index,element in enumerate(all_features['image_id'].tolist()):
                group=f"group_{group_number+index+1}"
                grouped_url_array[group] = []
                grouped_url_array[group].append(all_features.iloc[index+1]['image_id'])
        print(f"Groups formed are :: {grouped_url_array}")
        # #################################################################### Selection of Images from the groups ###################################################################################################################################################
        images = []
        print(f"Length of all groups :: {len(list(grouped_url_array.keys()))}")
        for key, val in grouped_url_array.items():
            print(f"Total images in the {key} :: {len(val)}")
            images.append(selection_from_groups(val, drive, job_uid))
        images=list(unpack(images)) 

        # #################################################################### Sorting The Images ###################################################################################################################################################
        print("################################################################################################################")
        print(f"Final length of selected images :::::::: {len(images)}")
        print("################################################################################################################")
        
        #image_urls = [url_image_id_mapper[id] for id in images]
        final_selection=selection(images, job_uid)   
        print("################################################################################################################")
        print(f"Final length of sorted df :::::::: {len(final_selection)}")
        print("################################################################################################################")

        final_selection = final_selection[0:int(number_of_output_images)]
        final_image_ids = final_selection['image_id'].tolist()

        folder = drive.CreateFile({
            "title"   : f"{folder_name}_fs_results", 
            "mimeType": "application/vnd.google-apps.folder"
            })

        folder.Upload()
        destination_folder = drive.ListFile({"q": "mimeType='application/vnd.google-apps.folder' and title='Bhushan_2021_01_25_fs_results' and trashed=false"}).GetList()[0]


        new_parent = destination_folder['id']
        files = drive.auth.service.files()

        for file_id in final_image_ids:
            file  = files.get(fileId= file_id, fields= 'parents').execute()
            prev_parents = ','.join(p['id'] for p in file.get('parents'))
            file  = files.update( fileId = file_id,
                                addParents = new_parent,
                                removeParents = prev_parents,
                                fields = 'id, parents',
                                ).execute()
        
        destination_folder.InsertPermission({
            'type': 'user',
            'value': 'pooja@pical.in',
            'role': 'reader'
            })

        # ############################################################ Creating the output  ###################################################################################################################################################

        # final_selection['image_title'] = final_selection.apply(lambda row: url_image_id_mapper[row['image_id']], axis=1)
        # current_time = time.time()
        # result = {"image_processing_selection_sorting_process" : {
        #     "output_raw_json" : str({
        #         'number_of_output_images' : number_of_output_images,
        #         'output_image_urls' : final_selection['image_title'].tolist(),
        #         'images_without_people' : images_without_people,
        #         'sns version' : 'v1.0',
        #         'timestamp' : current_time }),
        #     "status" : "success"
        # }}

        # print("hello world")
        # url = f"https://webhook.site/7a79bcdd-8ab9-4068-ae45-7fc9e43cf0cf"
        # x = requests.patch(url, json=result)

    """Return first the response and tie the consolidated_score to a thread"""
    Thread(target = consolidated_score).start()
    return jsonify({"status" : "ongoing", "message" : "We have Received the request. Please wait for 5 minutes wfor Job to complete"})
    
if __name__ == '__main__':
    app.run(debug=True, port=8877, host = '0.0.0.0')

