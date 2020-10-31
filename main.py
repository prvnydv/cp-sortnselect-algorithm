from dublicate import *
from face_extraction import img_to_faces
from utils import get_date_taken,unpack
from selection_from_groups import selection_from_groups
from final_selection import selection
from image_frequency import img_frequency
import time
from similarity import get_colors,color_diff
import cv2
import argparse
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-n", "--number", required=True,
	help="number of images in output")
args = vars(ap.parse_args())

image_dir=args['input']
face_dir= 'face_image'
new_img_dir='output'

start_time = time.time()


##################################################################### Removing All files except images ###################################################################################################################################################


path= image_dir+'/'
files = os.listdir(path)
for file in files:
    _,ext=file.split(".")
    if ext not in ['jpg','JPG','jpeg','JPEG','png','PNG']:
    	os.remove(path+file)



############################################################################ Dublication Removal ###################################################################################################################################################


for i in range(2):
	# 16 --> Hash Length
	# 240 --> Threshold
    remove_similar_from_dir(image_dir,16,240)
    os.remove('img_hashes_16.csv')


##################################################################### Removing All Images except images that have faces in them ###################################################################################################################################################


os.makedirs(face_dir)
path= image_dir+'/'
files = os.listdir(path)
for file in files:
    _,ext=file.split(".")
    if ext in ['jpg','JPG','jpeg','JPEG','png','PNG']:
        img_to_faces(path,file)   

image_id=[]
face_path=face_dir+'/'
files=os.listdir(face_path)
for file in files:
    name=[]
    name=file.split("$")
    image_id.append(name[1])
image_id=set(image_id)

files=os.listdir(path)
for i in range(len(files)):
    if files[i] not in image_id:
        os.remove(path+files[i])
shutil.rmtree(face_dir)    


##################################################################### Sorting Images based on timestamp ###################################################################################################################################################


files=os.listdir(image_dir)
date=[]
image_id=[]
for i in range(len(files)):
    date.append(get_date_taken(image_dir+'/'+files[i]))
    image_id.append(files[i])
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

faces=[]
count=[]
for i in range(len(a_list)):
    faces.append(a_list[i])
    count.append(len(a[a_list[i]]['images']))

image_cloud= list(zip(faces,count))
image_cloud = pd.DataFrame(image_cloud, columns = ['faces','count'])
image_cloud.to_csv('image_cloud.csv')


print("--- %s seconds ---" % (time.time() - start_time))