
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

app = Flask(__name__)



@app.route("/sns/v1", methods = ['POST'])
def consolidated_score():
    url_array = request.json['url_array']
    number_of_output_images = request.json['number_of_output_images']
    job_uid = request.json['job_uid']
    date_format = "%Y-%m-%d %H:%M:%S"
    current_time = datetime.strftime(datetime.now(), date_format)
    

    start_time = time.time()


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


    os.makedirs(face_dir)
    for url in url_array:
        ext = ('jpg','JPG','jpeg','JPEG','png','PNG')
        if url.endswith(ext):
            img_to_faces(job_uid, url)   



    image_id=[]
    face_path=face_dir+'/'
    face_files_url=s3_client.ls(f's3://sns-outputs/faces_extracted/{job_uid}')
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
        image_id.append(url)
    all_features= list(zip(image_id,date))
    all_features=pd.DataFrame(all_features)
    all_features.columns=["image_id","date_time"]
    all_features = all_features.sort_values(by=['date_time'], ascending=True)

    # #################################################################### Grouping Images based on Color Palette  ###################################################################################################################################################


    group_number=1
    group='group_test'
    os.makedirs(group)
    os.makedirs(group+'/'+str(group_number))
    img=cv2.imread(image_dir+'/'+all_features.iloc[0]['image_id'])
    cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[0]['image_id'],img)
    for i in range(len(all_features)-3):
        a,b,c,d=get_colors(image_dir+'/'+all_features.iloc[i]['image_id']),get_colors(image_dir+'/'+all_features.iloc[i+1]['image_id']),get_colors(image_dir+'/'+all_features.iloc[i+2]['image_id']),get_colors(image_dir+'/'+all_features.iloc[i+3]['image_id'])
        # Checking ith and (i+3)th image
        if color_diff(a,d)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+1]['image_id'],img)
                
                img2=cv2.imread(image_dir+'/'+all_features.iloc[i+2]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+2]['image_id'],img2)
                
                img3=cv2.imread(image_dir+'/'+all_features.iloc[i+3]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+3]['image_id'],img3)
                i+=2
            except:
                pass
        # Checking ith and (i+2)th image
        elif color_diff(a,c)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+1]['image_id'],img)
                
                img2=cv2.imread(image_dir+'/'+all_features.iloc[i+2]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+2]['image_id'],img2)
                i+=1
            except:
                pass
        # Checking ith and (i+1)th image
        elif color_diff(a,b)>5: # 6 out of 10 colors should be same if they are to be in same group
            try:
                img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+1]['image_id'],img)
            except:
                pass 
        else: # New group created if none of the above criterion are met
            group_number+=1
            os.makedirs(group+'/'+str(group_number))
            try:
                img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
                cv2.imwrite(group+'/'+str(group_number)+'/'+all_features.iloc[i+1]['image_id'],img)
            except:
                pass

    # #################################################################### Selection of Images from the groups ###################################################################################################################################################

    images=[]
    var='group_test/'
    group=os.listdir(var)
    for i in range(len(group)):
        images.append(selection_from_groups(var+group[i]))
    images=list(unpack(images)) 

    # #################################################################### Sorting The Images ###################################################################################################################################################


    final_selection=selection(image_dir,images)[:int(args['number'])]
    shutil.rmtree(var)    


    # ############################################################ Creating the output  ###################################################################################################################################################


    os.makedirs(new_img_dir)
    for i in range(len(final_selection)):
        try:
            img=cv2.imread(image_dir+'/'+final_selection.iloc[i]['image_id'])
            cv2.imwrite(new_img_dir+'/'+final_selection.iloc[i]['image_id'],img)
        except:
            pass

    # #################################################################### Creating the Face Count for Image Cloud ###################################################################################################################################################
    a=dict()
    os.makedirs(face_dir)
    final_selection=os.listdir(new_img_dir)
    path= image_dir+'/'
    for i in range(len(final_selection)):
        img_frequency(path,final_selection[i],a,i)
    a_list=list(a.keys())
    for i in range(len(a_list)):
        a[a_list[i]]['images']=set(a[a_list[i]]['images'])



    print("--- %s seconds ---" % (time.time() - start_time))


    result = {
      "input_image_urls" : url_array,
      "number_of_input_images" : len(url_array),
      "number_of_output_images" : number_of_output_images,
      "sns version" : "v1.0",
      "timestamp" : current_time
    }

    print("hello world")

    return jsonify(result=result)

    
if __name__ == '__main__':
    app.run(debug=True, port=8877, host = '0.0.0.0')

