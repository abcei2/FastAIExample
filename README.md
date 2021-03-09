# Custom_Train_MaskRCNN

Training on custom dataset with (multi/unique class) of a Mask RCNN

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
  **CUDA 11.0**
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
**REF:**https://pytorch.org/get-started/previous-versions/


### Structure
- dataset: folder where you put the train and val folders (read inside to know what to put).
- logs: folder where we store the intermediate/checkpoints and final weights after training
- weights: weights for the model (.h5 file), we fetch the weights from here for the test script
- weights_db: tells if weights is coco or not, to drop some layers for new classes.
- train_data.py: main script for train the model with the data.
- test_data.py: script to test the model with camera.

### Usage 


### Hardware requeriments

#### Train: 

For training is need atleast 12 GB of GPU and 8 of ram memory.  Inference works with cpu, but is very slow.

#### Val: 

For test is need atleast 2 GB of GPU and 4 of ram memory.  Inference works with cpu, but is quite slow.

### Dataset

The folder where the images and annotations should be place:

dataset/{dataset_name}/  
--images/ 
----models/ 
----0001.jpg
----0002.jpg
----0003.jpg   
----...  
--labels/  
----0001P.jpg
----0002P.jpg
----0003P.jpg   
----...
  
Where **images** are the original frame image and **labels** the masked images, which each object maps a pixel level, ej; to train 3 objects like person, car and bike, each image must have respectively the gray level of; 1,2,3, and the rest of the image are the backgroud labeled as 0.


