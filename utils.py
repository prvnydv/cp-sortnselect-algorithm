import re
import boto3
import s3fs
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile
import cv2

def get_date_taken(s3_uri):
    s3 = initiate_s3_resource_instance()
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
    aws_access_key_id = ''
    aws_secret_access_key = ''
    region_name = 'ap-south-1'
    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, 
                              aws_secret_access_key=aws_secret_access_key, 
                              region_name=region_name)

    return s3

def df_to_s3(df, job_uid, op_name, bucket='sns-outputs'):
    aws_access_key_id = ''
    aws_secret_access_key = ''

    path = str(job_uid)
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    with fs.open(f's3://{bucket}/{path}/{op_name}.csv', 'wb') as f:
        f.write(bytes_to_write)

def df_from_s3(job_uid, op_name):
    aws_access_key_id = ''
    aws_secret_access_key = ''

    fs = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    key = f'{job_uid}/{op_name}.csv'
    bucket = 'sns-outputs'

    df = pd.read_csv(fs.open(f'{bucket}/{key}', mode='rb'))

    return df

def parse_bucket_key(s3_uri):
    parse_url = urlparse(s3_uri, allow_fragments = False)
    bucket = parse_url.netloc
    key = parse_url.path.lstrip('/')

    return bucket, key

def read_pillow_image_from_s3(s3_uri):
    s3 = initiate_s3_resource_instance()
    bucket, key = parse_bucket_key(s3_uri)
    file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    return Image.open(BytesIO(file_byte_string))

def read_with_cv2_from_generated_temp_file(s3_uri):
    s3 = initiate_s3_resource_instance()
    bucket, key = parse_bucket_key(s3_uri)

    object = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        object.download_fileobj(f)
        img=cv2.imread(tmp.name)

        f.close()
    
    return img

def write_cv2_image_to_s3(image, folder_name, file_name, job_uid, bucket='sns-outputs'):
    s3 = initiate_s3_resource_instance()
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(tmp.name, image)
    
    with open(tmp.name, 'rb') as f:
        s3.Bucket(bucket).put_object(Key= f'{folder_name}/{job_uid}/{file_name}.jpg', Body=f, ContentType= 'image/png')
        f.close()


def s3_client():
    aws_access_key_id = ''
    aws_secret_access_key = ''
    return s3fs.S3FileSystem(anon=False,
                             key=aws_access_key_id,
                             secret=aws_secret_access_key))