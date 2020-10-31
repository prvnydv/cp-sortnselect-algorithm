import numpy as np
from PIL import Image, ImageDraw

def get_colors(image_file, numcolors=10, resize=150):
    # Resize image to speed up processing
    img = Image.open(image_file)
    img = img.copy()
    img.thumbnail((resize, resize))
    # Reduce to palette
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=numcolors)
    # Find dominant colors
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    colors = list()
    for i in range(numcolors):
        palette_index = color_counts[i][1]
        dominant_color = palette[palette_index*3:palette_index*3+3]
        colors.append(np.array(dominant_color))
    return colors

# Finding color similarity between images on their 10-color palette
def color_diff(a,b):
    c=0
    for i in range(10):
        for  j in range(10):
            if abs(a[i]-b[j]).sum()<40:
                c+=1
                break
            else:
                continue
    return c