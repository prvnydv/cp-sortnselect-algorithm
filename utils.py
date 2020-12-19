import re
import boto3
import s3fs
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import tempfile
import cv2
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def get_date_taken(s3_uri):
    s3 = initiate_s3_client_instance()
    bucket, key = parse_bucket_key(s3_uri)
    file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()

    date_time=re.split(':| ',Image.open(BytesIO(file_byte_string))._getexif()[36867])
    return int(date_time[0]+date_time[1]+date_time[2]+date_time[3]+date_time[4]+date_time[5])

# unpacking [1,[2,3],{4,5,6}] --> [1,2,3,4,5,6]
def unpack(seq):
    if isinstance(seq, (list, tuple, set)):
        yield from (x for y in seq for x in unpack(y))
    elif isinstance(seq, dict):
        yield from (x for item in seq.items() for y in item for x in unpack(y))
    else:
        yield seq

def initiate_s3_resource_instance():
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'
    region_name = 'ap-south-1'
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, 
                              aws_secret_access_key=aws_secret_access_key, 
                              region_name=region_name)

    return s3

def initiate_s3_client_instance():
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'
    region_name = 'ap-south-1'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                              aws_secret_access_key=aws_secret_access_key, 
                              region_name=region_name)

    return s3

def df_to_s3(df, job_uid, op_name, bucket='pical-backend-dev'):
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'

    path = str(job_uid)
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(anon=False, key=aws_access_key_id, secret=aws_secret_access_key, use_ssl=False)
    with fs.open(f"s3://{bucket}/{path}/{op_name}", 'wb') as f:
        f.write(bytes_to_write)

def df_from_s3(job_uid, op_name):
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'

    fs = s3fs.S3FileSystem(anon=False, key=aws_access_key_id, secret=aws_secret_access_key, use_ssl=False)
    key = f"{job_uid}/{op_name}.csv"
    bucket = 'pical-backend-dev'

    df = pd.read_csv(fs.open(f"{bucket}/{key}", mode='rb'))

    return df

def parse_bucket_key(s3_uri):
    parse_url = urlparse(s3_uri, allow_fragments = False)
    bucket = parse_url.netloc
    key = parse_url.path.lstrip('/')

    return bucket, key

def read_pillow_image_from_s3(s3_uri):
    s3 = initiate_s3_client_instance()
    bucket, key = parse_bucket_key(s3_uri)
    file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    return Image.open(BytesIO(file_byte_string))

def read_with_cv2_from_generated_temp_file(s3_uri):
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'
    region_name = 'ap-south-1'

    s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key, 
                      region_name = region_name)

    bucket, key = parse_bucket_key(s3_uri)

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        s3.download_fileobj(bucket, key, f)
        img=cv2.imread(tmp.name)

        f.close()
    
    return img

def write_cv2_image_to_s3(image, folder_name, file_name, job_uid, bucket='pical-backend-dev'):
    print(f'Writing Images int o s3')
    s3 = initiate_s3_resource_instance()
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(tmp.name, image)
    
    with open(tmp.name, 'rb') as f:
        s3.Bucket(bucket).put_object(Key= f"{job_uid}/{folder_name}/{file_name}", Body=f, ContentType= 'image/png')
        f.close()


def s3_client():
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'
    return s3fs.S3FileSystem(anon=False,
                             key=aws_access_key_id,
                             secret=aws_secret_access_key,
                             use_ssl=False)

def check_if_file_present(bucket, key):
    s3_resource = initiate_s3_resource_instance()
    bucket = s3_resource.Bucket(bucket)
    key = key
    objs = list(bucket.objects.filter(Prefix=key))
    if len(objs) > 0 and objs[0].key == key:
        return True
    else:
        return False

def list_all_objects_of_a_bucket_folder(bucket_name:str, folder_name:str):
    s3 = initiate_s3_resource_instance()
    bucket = s3.Bucket(bucket_name)

    s3_uris = []
    files_in_s3 = bucket.objects.all()
    for file in list(files_in_s3):
        if folder_name in file.key:
            s3_uris.append(f's3://{bucket_name}/{file.key}')
    
    return s3_uris

def list_key_bucket_object(bucket_name:str, folder_name:str):
    s3 = initiate_s3_resource_instance()
    bucket = s3.Bucket(bucket_name)

    s3_uris = []
    files_in_s3 = bucket.objects.all()
    for file in list(files_in_s3):
        if folder_name in file.key:
            s3_uris.append(file)

    return s3_uris

def load_image_for_keras(s3_uri, target_size):
    aws_access_key_id = 'AKIAJRZVZ6HMUSSYSXYQ'
    aws_secret_access_key = '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'
    region_name = 'ap-south-1'

    s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key, 
                      region_name = region_name)

    bucket, key = parse_bucket_key(s3_uri)

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        s3.download_fileobj(bucket, key, f)
        img=load_img(tmp.name, target_size=target_size)

        f.close()
    
    return img


