from fastai.vision import *
from fastai.callbacks.hooks import *
import os
import shutil
#URLs.CAMVID Download camvid database, save and untar on folder data
path = untar_data(URLs.CAMVID,dest="./dataset/")
path_lbl = path/'labels'
path_img = path/'images'
#CLASS LABELS
codes = np.loadtxt(path/'codes.txt', dtype=str)
#LAMNDA TO LOAD DATA
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
fnames = get_image_files(path_img)
mask = open_image(get_y_fn(fnames[0]))
src_size = np.array(mask.shape[1:])
#SIZE OF TRAINING IMAGE AND BATCH SIZE
size = src_size//4
bs = 2

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
    .databunch(bs=bs)
    .normalize(imagenet_stats))

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
    

metrics = acc_camvid
wd = 1e-2

# create unet
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd) 
learn.lr_find()
learn.recorder.plot()

lr = 3e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

learn.save('camvid-stage-save')
learn.export('camvid-stage-1')

learn.unfreeze()
learn.lr_find() 

learn.recorder.plot()   
learn.fit_one_cycle(12, slice(lr/400, lr/4), pct_start=0.8)
learn.show_results(rows=3, figsize=(10, 10))

learn.save('camvid-stage-2-save')
learn.export('camvid-stage-2')