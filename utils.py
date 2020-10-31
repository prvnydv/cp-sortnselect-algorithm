from PIL import Image
import re

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