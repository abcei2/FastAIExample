import cv2
import glob
import numpy as np
import re
from pathlib import Path
# from fastai.vision import *
# from fastai.callbacks.hooks import *


masks=glob.glob('./data/pavimento/mask_image/*.png')
images=glob.glob('./data/pavimento/original_images/*.png')
#print(masks)
pixel_classes=[  0,  19,  29,  38,  57,  79, 136, 147, 155, 167, 217, 255]
class_count=[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

uniques=[]
for image in images:
    mask=image[0:-4]+"GT.png"
    mask_gray=image[0:-4]+"GT.png"
    	
    mask = re.sub(r'\b' + 'original_images' + r'\b', 'mask_image', mask)
    mask_gray = re.sub(r'\b' + 'original_images' + r'\b', 'labels', mask_gray)
    
    mask_image=cv2.imread(mask)
    original_image=cv2.imread(image)

    mask_image_gray=cv2.cvtColor(mask_image,cv2.COLOR_RGB2GRAY)   
    # mask_image_gray=original_image.copy()
    mask_one_object=np.zeros(mask_image_gray.shape)

    for ii in range(12):
        mask_one_object=np.zeros(mask_image_gray.shape)
        posit=np.argwhere(np.array(mask_image_gray)==pixel_classes[ii])
        
        for kk in range(len(posit)):
            mask_one_object[posit[kk][0]][posit[kk][1]]=255
        if len(posit)>0:
            alpha=0.5
            mask_one_object=np.dstack([mask_one_object,mask_one_object,mask_one_object])
            print(mask_one_object.shape,original_image.shape)
            mask_one_object=cv2.addWeighted(mask_one_object, alpha, original_image, 1-alpha,0, dtype = cv2.CV_32F)
            Path(f"temporal/clase{ii}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"temporal/clase{ii}/clase{ii}_{class_count[ii]}.jpg",mask_one_object)
            class_count[ii]+=1

    
    uniques_aux=[]
    for i in np.unique(mask_one_object):
        uniques.append(i)
    

    # cv2.imshow("original_image",original_image)
    # # cv2.imshow("mask_image",mask_image*mask_one_object)
    # cv2.imshow("mask_image_gray",mask_image_gray*mask_one_object)
    # #cv2.imshow("mask_one_object",mask_one_object*255)
    # cv2.imwrite(mask_gray,mask_one_object)
    # cv2.waitKey(10)
    print(np.unique(uniques))
    print(class_count)
print(np.unique(uniques))