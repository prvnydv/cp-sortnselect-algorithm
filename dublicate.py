import imagehash
import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.neighbors import KDTree
from urllib.parse import urlparse
from utils import initiate_s3_resource_instance
from utils import df_to_s3, df_from_s3
import s3fs

def read_pillow_image_from_s3(s3_uri):
    s3 = initiate_s3_resource_instance()
    parse_url = urlparse(s3_uri, allow_fragments = False)
    bucket = parse_url.netloc
    key = parse_url.path.lstrip('/')
    file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    return Image.open(BytesIO(file_byte_string))


def img_hash(url, hash_size): # fetching image hashes   
    return imagehash.phash(read_pillow_image_from_s3(url),hash_size=hash_size)


def get_hashes(url_array, hash_size, job_uid): # Adding the hashes to csv for future use
    hash_file = f'img_hashes_#{hash_size}.csv'
    if not os.path.isfile(hash_file):
        hashes = pd.DataFrame()
    else:
        hashes = df_from_s3(job_uid, hash_file)
    new_hashes_calculated = 0
    num_of_files=len(url_array)
    for url in url_array:
        if 'file' not in hashes.columns or url not in list(hashes['file']):
        	try:                                             
	            new_hashes_calculated = new_hashes_calculated + 1
	            result = {'file': url,'hash':img_hash(url,hash_size)}
	            hashes = hashes.append(result,ignore_index=True)
	            if (new_hashes_calculated % 200 == 199):
                    df_to_s3(hashes[['file','hash']], job_uid, hash_file)
	        except:
	        	pass 
    if new_hashes_calculated:
        df_to_s3(hashes[['file','hash']], job_uid, hash_file)    
    return read_hashes(hash_size)


def read_hashes(hash_size, job_uid): # formatting the hash reading in csv
    hash_file = f'img_hashes_#{hash_size}.csv'
    hashes = df_from_s3(job_uid, hash_file)[['file','hash']]
    lambdafunc = lambda x: pd.Series([int(i,16) for key,i in zip(range(0,len(x['hash'])),x['hash'])])
    newcols = hashes.apply(lambdafunc, axis=1)
    newcols.columns = [str(i) for i in range(0,len(hashes.iloc[0]['hash']))]
    return hashes.join(newcols)


def remove_similar_from_dir(url_array,hash_size,threshold, job_uid):
    hashes_16_lag = get_hashes(url_array,hash_size, job_uid)
    # Clustering the images based on image hashes using KDTree Algorithm
    t = KDTree(hashes_16_lag[[str(i) for i in range(0,64)]],metric='manhattan')
    distances, indices = t.query(hashes_16_lag[[str(i) for i in range(0,64)]],k=2)
    indices_pairs_of_closest_distance=[]
    for i in range(len(distances)):
        if distances[i][1]<threshold: # checking distances of image hashes to determine if the are same
            indices_pairs_of_closest_distance.append(indices[i])
    unique_pairs = [pair for pair in indices_pairs_of_closest_distance if (pair == np.sort(pair)).all()]
    similar=[]
    for i in range(len(unique_pairs)):
        similar.append(unique_pairs[i][1])
    keep=[]
    for i in range(len(unique_pairs)):
        keep.append(unique_pairs[i][0])
    rem=similar.copy()
    for i in range(len(keep)):
        if keep[i] in rem:
            similar.remove(rem[i])
    col_dir = directory+'*.jpg'
    col = os.listdir(directory)
    index=[]
    for i in range(len(col)):
        index.append(i)
    dictionary=dict(zip(index,col))
    for i in similar:
        try:
            path = os.path.join(directory, dictionary[i])
            os.remove(path)
        except OSError:
            pass   