# FastAIExample

Training on custom dataset with Unet+Resnet using FastAI library

### Requirements 
Specific version
```
    python3
    numpy==1.19.5
    opencv-python==4.5.1.48
    torch==1.7.1
    torchvision==0.8.2
    scikit-image==0.17.2
    fastai==1.0.61
    cuda==11.0  (?)
```

Installation

```
  sudo apt-get install python3 python3-pip
  pip install -r requeriments.txt
  #CUDA 11.0
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
**REF:**https://pytorch.org/get-started/previous-versions/


### Structure
- dataset: folder where you put the images and labels folders (read inside to know what to put).
- weights_db: tells if weights is coco or not, to drop some layers for new classes.
- train_model.py: main script for train the model with the data.
- test_model.py: script to test the model with camera.

### Usage 

#### Train: 

On train_model.py we have this main path and variables.  

```
path = untar_data(URLs.CAMVID,dest="./dataset/")
path_lbl = path/'labels'
path_img = path/'images'
#CLASS LABELS
codes = np.loadtxt(path/'codes.txt', dtype=str)
#LAMNDA TO LOAD DATA
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

```
**path:** Must have path to the dataset, the basic example uses CAMVID dataset.  
**path_img:** path to original images in a dataset.  
**path_lbl:** path to labels in a dataset. Labels are the masked images.  
**codes:**  path to *codes.txt* which have all class names.  
**get_y_fn:** lambda function that maps, path_img images with path_lbl mask images.  

```
python train_model.py
```
### Hardware requeriments

#### Train: 

For training is need atleast 8 GB of GPU and 8 of ram memory (depending on data image size).  Train works with cpu, but is very slow.

#### Val: 

For test is need atleast 2 GB of GPU and 4 of ram memory.  Inference works with cpu, but is quite slow.

### Dataset

The folder where the images and labels should be place:
**IMAGES MUST BE IN PNG FORMAT!!! VERY IMPORTANT**

dataset/{dataset_name}/  
--images/  
----models/   
----0001.png  
----0002.png  
----0003.png    
----...  
--labels/  
----0001P.png  
----0002P.png  
----0003P.png   
----...  
  
Where **images** are the original frame image and **labels** the masked images, which each object maps a pixel level, ej; to train 3 objects like person, car and bike, each image must have respectively the gray level of; 1,2,3, and the rest of the image are the backgroud labeled as 0.


