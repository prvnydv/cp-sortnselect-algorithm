import imagehash
import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.neighbors import KDTree
from urllib.parse import urlparse
from utils import initiate_s3_resource_instance

def read_pillow_image_from_s3(s3_uri):
    s3 = initiate_s3_resource_instance()
    parse_url = urlparse(s3_uri, allow_fragments = False)
    bucket = parse_url.netloc
    key = parse_url.path.lstrip('/')
    file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    return Image.open(BytesIO(file_byte_string))


def img_hash(url, hash_size): # fetching image hashes   
    return imagehash.phash(read_pillow_image_from_s3(url),hash_size=hash_size)

    if not os.path.isfile(hash_file):
        hashes = pd.DataFrame()
    else:
        hashes = pd.read_csv(hash_file)
    new_hashes_calculated = 0
    num_of_files=len(os.listdir(directory))
    for file in os.listdir(directory):
        if 'file' not in hashes.columns or file not in list(hashes['file']):
        	try:                                             
	            new_hashes_calculated = new_hashes_calculated + 1
	            result = {'file': file,'hash':img_hash(directory  +'/'+ file,hash_size)}
	            hashes = hashes.append(result,ignore_index=True)
	            if (new_hashes_calculated % 200 == 199):
	                hashes[['file','hash']].to_csv(hash_file,index=False)
	        except:
	        	pass 
    if new_hashes_calculated:
        hashes[['file','hash']].to_csv(hash_file,index=False)    
    return read_hashes(hash_size)


def read_hashes(hash_size): # formatting the hash reading in csv
    hash_file = 'img_hashes_%s.csv' % hash_size
    hashes = pd.read_csv(hash_file)[['file','hash']]
    lambdafunc = lambda x: pd.Series([int(i,16) for key,i in zip(range(0,len(x['hash'])),x['hash'])])
    newcols = hashes.apply(lambdafunc, axis=1)
    newcols.columns = [str(i) for i in range(0,len(hashes.iloc[0]['hash']))]
    return hashes.join(newcols)


def remove_similar_from_dir(directory,hash_size,threshold):
    hashes_16_lag = get_hashes(directory,hash_size)
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