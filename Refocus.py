import cv2
import numpy as np
from pathlib import Path
def refocus(Light_field, patch_len, factor = 1, a = 1):
    '''
    factor : the factor that used to scale the image
    '''
    size = (int(Light_field[0, 0].shape[1]*factor),int(Light_field[0, 0].shape[0]*factor))
    overlap = int(abs(a) * (patch_len - 1))
    img_slice = np.zeros((size[1]+overlap,size[0]+overlap,3))
    for j in range(patch_len):
        for i in range(patch_len):
            # perform sub-pixel refinement if required
            if factor != 1:
                vp_img = cv2.resize(Light_field[j, i],size) / (patch_len **2)
            else:
                vp_img = Light_field[j, i] / 25 #(patch_len ** 2) 
                
                
            # get viewpoint padding for each border
            tb = (int(abs(a) * j), int(abs(a) * (patch_len - 1 - j)))    # top, bottom
            lr = (int(abs(a) * i), int(abs(a) * (patch_len - 1 - i)))    # left, right

            # flip padding for each axis if a is negative
            pad_width = (tb, lr, (0, 0)) if a >= 0 else (tb[::-1], lr[::-1], (0, 0))
            # shift viewpoint image and add its values to refocused image slice
            img_slice = np.add(img_slice, np.pad(vp_img, pad_width, 'edge'))
            # print(img_slice)

    # crop refocused image for consistent image dimensions
    img_slice=np.array(img_slice)
    crop = int(overlap/2)
    final_img = img_slice[crop:-crop, crop:-crop, :] if (a != 0) else img_slice
    final_img = cv2.resize(img_slice,(Light_field[0, 0].shape[1],Light_field[0, 0].shape[0]))

    return final_img
