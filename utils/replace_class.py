import cv2
import glob
import numpy as np
import re
from pathlib import Path


masks=glob.glob('./camvid/labels/*.png')
images=glob.glob('./camvid/images/*.png')
new_dbname="temporal/camvidnew"
images_folder=new_dbname + "/images"
labels_folder=new_dbname + "/labels"
counter=0

Path(new_dbname).mkdir(parents=True, exist_ok=True)
Path(images_folder).mkdir(parents=True, exist_ok=True)
Path(labels_folder).mkdir(parents=True, exist_ok=True)
#print(masks)
uniques=[]
pixel_classes=[2, 4, 5, 8, 10, 16, 17, 19, 20, 24]
for image in images:
    mask=image[0:-4]+"_P.png"
    mask_gray=image[0:-4]+"_P.png"
    print(image)
    mask = re.sub(r'\b' + 'images' + r'\b', 'labels', mask)
    mask_gray = re.sub(r'\b' + 'images' + r'\b', 'labels', mask_gray)
    mask_image=cv2.imread(mask)
    print(mask)
    original_image=cv2.imread(image)

    mask_image_gray=mask_image.copy()   

    mask_one_object=np.zeros(mask_image_gray.shape)

    for ii in range(len(pixel_classes)):
        
        posit=np.argwhere(np.array(mask_image_gray)==pixel_classes[ii])
        print(f"pixel {ii}")
        for kk in range(len(posit)):
            mask_one_object[posit[kk][0]][posit[kk][1]]=ii+1
    print(f"max pixel value {np.max(mask_one_object)}")

    
    cv2.imwrite(f"{labels_folder}/temporal{counter}_P.png",mask_one_object)
    cv2.imwrite(f"{images_folder}/temporal{counter}.png",original_image)
    counter+=1