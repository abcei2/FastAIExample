import cv2
import glob
import numpy as np
import re
from pathlib import Path


masks=glob.glob('./temporal/camvidnew/labels/*.png')
images=glob.glob('./temporal/camvidnew/images/*.png')
# print(images)
indexes=np.random.choice(len(images), 100, replace=True)
indexes=[int(intIndex) for intIndex in indexes]
print(indexes)
counter=0
# print(np.random.choice(len(images), 100, replace=True))
with open("valid.txt","w") as f:  
    for index in indexes:
        # print(images[index].split("/")[4])
        f.write(images[index].split("/")[4]+"\n") 