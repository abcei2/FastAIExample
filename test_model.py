
from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
import cv2
from skimage import color
torch.backends.cudnn.benchmark=True


path='data/camvid/'
path_lbl = path+'labels/'
path_img = path+'images/'
get_y_fn = lambda x: path_lbl+f'{x.stem}_P{x.suffix}'

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)

img_f = fnames[0]
mask = open_mask(get_y_fn(img_f))
src_size = np.array(mask.shape[1:])

codes = np.loadtxt(path+'codes.txt', dtype=str)

size = src_size//2
bs = 2

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
.databunch(bs=bs)
.normalize(imagenet_stats))


cap=cv2.VideoCapture(2)
if __name__ == '__main__':
    
    #learn = load_learner('data/camvid/camvid-stage-save/archive')
    learn = unet_learner(data,models.resnet34)
    learn.load('../../camvid-stage-save')
    plt.figure()
    colors=np.random.random((12, 3))
    print(colors)
    frame_skipping=0
    while True:
        ret, frame=cap.read()
        frame_skipping+=1
        if frame_skipping%1==0:
            frame_skipping=0
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=cv2.resize(frame,(480 ,360))
            cv2.imwrite("frame.jpg",frame)
            img= open_image("frame.jpg")

            prediction=learn.predict(img)

            predict_mask1=prediction[1].cpu().detach().numpy()            
            mask_one_object=np.zeros(frame.shape)
            
            print(np.min(predict_mask1),len(codes))
            for ii in range(len(codes)):
            
                posit=np.argwhere(predict_mask1[0]==ii)
                
                for kk in range(len(posit)):
                    mask_one_object[posit[kk][0]][posit[kk][1]]=int(255-255*(len(codes)-ii)/(len(codes)))

            print(type(mask_one_object),type(frame),mask_one_object.shape,frame.shape)
            mask_one_object=np.asarray(mask_one_object,dtype="uint8")

            zero_frames=np.zeros(frame.shape)
            frame_mask=cv2.resize(color.label2rgb(predict_mask1[0],zero_frames,bg_label=0,colors=colors,kind='overlay'),(800 ,600))
            cv2.imshow("mask_image_one",mask_one_object)
        cv2.imshow("original_image",frame)
        cv2.waitKey(10)