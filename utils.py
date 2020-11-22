import re
import boto3
import s3fs
from PIL import Image
from io import BytesIO


def get_date_taken(path):
    date_time=re.split(':| ',Image.open(path)._getexif()[36867])
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
                                     aws_secret_access_key=aws_secret_access_key, 
                              aws_secret_access_key=aws_secret_access_key, 
                              region_name=region_name)

    return s3

def df_to_s3(df, job_uid, op_name):
    aws_access_key_id = ''
    aws_secret_access_key = ''

    bucket = 'sns-outputs'
    path = str(job_uid)
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    with fs.open(f's3://#{bucket}/#{path}/#{op_name}.csv', 'wb') as f:
        f.write(bytes_to_write)

def df_from_s3(job_uid, op_name):
    aws_access_key_id = ''
    aws_secret_access_key = ''

    fs = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    key = f'{str(job_uid)}\{op_name}.csv'
    bucket = 'sns-outputs'

    df = pd.read_csv(fs.open('{}/{}'.format(bucket, key),
                            mode='rb')
                    )
    return df

def read_pillow_image_from_s3(s3_uri):
    s3 = initiate_s3_resource_instance()
    parse_url = urlparse(s3_uri, allow_fragments = False)
    bucket = parse_url.netloc
    key = parse_url.path.lstrip('/')
    file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    
    return Image.open(BytesIO(file_byte_string))


