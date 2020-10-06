from dublicate import *
from face_extraction import img_to_faces
from happy import expression_image
from eyes import eyes_dir
from age import face_age
from gender import gender_pred
from date_time import get_date_taken
import time
from similarity import dominant
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())

image_dir=args['input']
face_dir= 'faces'
new_img_dir=args['output']


start_time = time.time()

#################################################################### Dublication Removal ###################################################################################################################################################


for i in range(2):
    remove_similar_from_dir(image_dir,16,240)
    os.remove('img_hashes_16.csv')


##################################################################### Face Extraction ###################################################################################################################################################


os.makedirs(face_dir)
path= image_dir+'/'
files = os.listdir(path)
for file in files:
    _,ext=file.split(".")
    if ext in ['jpg','JPG','jpeg','JPEG']:
        img_to_faces(path,file)


##################################################################### Emotion Prediction ###################################################################################################################################################


index=[]
image_id=[]
happy=[]
face_path=face_dir+'/'
files=os.listdir(face_path)
for file in files:
    name=[]
    name=file.split("_")
    happy.append(expression_image(face_path+file)[0])
    index.append(name[0])
    image_id.append(name[1])

data= list(zip(index,image_id,happy))
final_data = pd.DataFrame(data, columns = ['index', 'image_id','happy'])
grouped = final_data.groupby('image_id')
happy=pd.DataFrame(grouped['happy'].agg(np.mean))


#################################################################### Candid Prediction ###################################################################################################################################################


eyes_focus=[]
for file in files:
    name=[]
    name=file.split("_")
    eyes_focus.append(eyes_dir(face_path+file))
    
data= list(zip(index,image_id,eyes_focus))
final_data = pd.DataFrame(data, columns = ['index', 'image_id','not_candid'])
grouped = final_data.groupby('image_id')
eyes=grouped['not_candid'].agg(np.mean)
eyes=pd.DataFrame(eyes)


#################################################################### Gender Prediction ###################################################################################################################################################


gender=[]
for file in files:
    name=[]
    name=file.split("_")
    result=gender_pred(face_path+file)
    gender.append(result[0][0][0])
    

data= list(zip(index,image_id,gender))
final_data = pd.DataFrame(data, columns = ['index', 'image_id','gender'])
grouped = final_data.groupby('image_id')
gender=pd.DataFrame(grouped['gender'].agg(np.mean))


#################################################################### Age Prediction ###################################################################################################################################################


age=[]
for file in files:
    name=[]
    name=file.split("_")
    result=face_age(face_path+file)
    age.append(result)

data= list(zip(index,image_id,age))
final_data = pd.DataFrame(data, columns = ['index', 'image_id','age'])
grouped = final_data.groupby('image_id')
age=pd.DataFrame(grouped['age'].agg(np.mean))


############################################################ Aggregating Features and Date/time based Sorting  ###################################################################################################################################################


all_features2=[happy,eyes,age,gender]
all_features2=pd.concat(all_features2,axis=1)
all_features2['final_feature']=all_features2.apply(lambda row: row.gender*0.1 + 0.2/row.age+ row.not_candid*0.4+ row.happy*0.3, axis=1)
all_features2 = all_features2.sort_values(by=['final_feature'], ascending=False)
all_features2.reset_index(inplace = True)
all_features=all_features2[:10]
del all_features2

date=[]
for i in range(len(all_features)):
    date.append(get_date_taken(image_dir+'/'+all_features.iloc[i]['image_id']))
    
all_features['date_time'] = date
all_features = all_features.sort_values(by=['date_time'], ascending=True)


#################################################################### Nearness Removal ###################################################################################################################################################


os.makedirs(new_img_dir)
img=cv2.imread(image_dir+'/'+all_features.iloc[0]['image_id'])
cv2.imwrite(new_img_dir+'/'+all_features.iloc[0]['image_id'],img)

for i in range(len(all_features)-1):
    time_delay=abs(all_features.iloc[i]['date_time'] - all_features.iloc[i+1]['date_time'])
    if time_delay < 50:
        continue
    elif time_delay < 500:
        if abs((dominant(image_dir+'/'+all_features.iloc[i]['image_id'])- dominant(image_dir+'/'+all_features.iloc[i+1]['image_id'])).sum())<80:
            continue
        else:
            try:
                img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
                cv2.imwrite(new_img_dir+'/'+all_features.iloc[i+1]['image_id'],img)
            except:
                pass
    else:
        try:
            img=cv2.imread(image_dir+'/'+all_features.iloc[i+1]['image_id'])
            cv2.imwrite(new_img_dir+'/'+all_features.iloc[i+1]['image_id'],img)
        except:
            pass


print("--- %s seconds ---" % (time.time() - start_time))