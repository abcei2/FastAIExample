import cv2
import glob
import numpy as np
import re
from pathlib import Path
# from fastai.vision import *
# from fastai.callbacks.hooks import *



# masks=glob.glob('./data/pavimento/mask_image/*.png')
# images=glob.glob('./data/pavimento/original_images/*.png')
#print(masks)
#PAVIMENTO DB
# pixel_classes=[  0,  19,  29,  38,  57,  79, 136, 147, 155, 167, 217, 255]
# class_count=[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
#CAMVID
masks=glob.glob('./temporal/camvidnew/labels/*.png')
images=glob.glob('./temporal/camvidnew/images/*.png')
pixel_classes=[i for i in range(32)]
class_count=[ 0 for i in range(32) ]
print(masks)
uniques=[]
for image in images:
    mask=image[0:-4]+"_P.png"
    mask_gray=image[0:-4]+"_P.png"
    	
    mask = re.sub(r'\b' + 'images' + r'\b', 'labels', mask)
    mask_gray = re.sub(r'\b' + 'images' + r'\b', 'labels', mask_gray)
    
    mask_image=cv2.imread(mask,0)
    original_image=cv2.imread(image)

    # mask_image_gray=cv2.cvtColor(mask_image,cv2.COLOR_RGB2GRAY)   
    mask_image_gray=mask_image.copy()
    mask_one_object=np.zeros(mask_image_gray.shape)
    print("asfdaf",np.max(mask_image))
    for ii in range(len(class_count)):
     
        mask_one_object=np.zeros(mask_image_gray.shape)
        
        posit=np.argwhere(np.array(mask_image_gray)==pixel_classes[ii])
        
        for kk in range(len(posit)):
            mask_one_object[posit[kk][0]][posit[kk][1]]=255
        if len(posit)>0:
            alpha=0.5
            mask_one_object=np.dstack([mask_one_object,mask_one_object,mask_one_object])
            print(mask_one_object.shape,original_image.shape)
            mask_one_object=cv2.addWeighted(mask_one_object, alpha, original_image, 1-alpha,0, dtype = cv2.CV_32F)
            Path(f"temporal2/clase{ii}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"temporal2/clase{ii}/clase{ii}_{class_count[ii]}.png",mask_one_object)
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