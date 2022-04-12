import cv2
import numpy as np
from pathlib import Path
def refocus(Light_field, num_views, factor = 1, a = 1):
    '''
    Light_field : light field images array (It should contain num_views*num_views images)
    num_views : the number of views each rows or columns
    factor : the parameter which is used to scale the image
    a : focal length (It can be negative)
    '''
    size = (int(Light_field[0, 0].shape[1]*factor),int(Light_field[0, 0].shape[0]*factor))
    overlap = int(abs(a) * (num_views - 1))
    img_slice = np.zeros((size[1]+overlap,size[0]+overlap,3))
    for j in range(num_views):
        for i in range(num_views):
            # Because we are going to add all image together, each image should be divide by num_views ** 2
            if factor != 1:
                vp_img = cv2.resize(Light_field[j, i],size) / (num_views **2)
            else:
                vp_img = Light_field[j, i] / (num_views ** 2) 
                
            # get viewpoint padding for each border
            tb = (int(abs(a) * j), int(abs(a) * (num_views - 1 - j)))    # top, bottom
            lr = (int(abs(a) * i), int(abs(a) * (num_views - 1 - i)))    # left, right

            # flip padding for each axis if a is negative
            pad_width = (tb, lr, (0, 0)) if a >= 0 else (tb[::-1], lr[::-1], (0, 0))
            # shift viewpoint image and add its values to refocused image slice
            img_slice = np.add(img_slice, np.pad(vp_img, pad_width, 'edge'))

    # crop refocused image for consistent image dimensions
    crop = int(overlap/2)
    final_img = img_slice[crop:-crop, crop:-crop, :] if (a != 0) else img_slice
    final_img = cv2.resize(img_slice,(Light_field[0, 0].shape[1],Light_field[0, 0].shape[0]))
    final_img = final_img.astype(np.uint8)

    return final_img
