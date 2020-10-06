from PIL import Image
import re

def get_date_taken(path):
    date_time=re.split(':| ',Image.open(path)._getexif()[36867])
    return int(date_time[0]+date_time[1]+date_time[2]+date_time[3]+date_time[4]+date_time[5])