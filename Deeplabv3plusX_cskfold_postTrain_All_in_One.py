# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:12:38 2022
1. This script is modified from RGB version, to make the input images all convert to Gray type, modified at Sept 29, 2022
2. save the state_dict(), not the whole model, since if may cause some problem in different platform
3. all input images convert to Gray, Oct 2, 2022
4. use golbal variables, Oct 2, 2022
5. Oct 7, try use resnet50
6. Nov 17, use new dataset to post train the model,to see the performace
Tips:
    if run on Windows, num_worker=0, if sockeye, num_worker=6*gpus
@author: Wenqi
"""
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, KFold, StratifiedKFold
from datetime import datetime
import time
import ntpath
import shutil
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
from glob import glob
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
import argparse
# import logging
import socket

# logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d/ %H:%M:%S ')
# logging.basicConfig(level=logging.INFO,file='logfile.log')

# torch.hub.set_dir('/scratch/st-marwanh-1/wenqi94/')

parser=argparse.ArgumentParser()
parser.add_argument("--data_dir",type=str,default='./data_RGB')
parser.add_argument("--sub_dataset",type=str,default='Rongxuehe_2_50mask')
parser.add_argument("--EPOCHS",type=int,default=10)
parser.add_argument("--Batch_size",type=int,default=6)
parser.add_argument("--FoldNumber",type=int,default=5)
parser.add_argument("--Force_to_Gray",type=bool,default=True)
parser.add_argument("--Encoder",type=str,default='resnet152')
args=parser.parse_args()
# logging.info(args)

# print(args.data_dir)
## get the images and labels directory
def listdir_nohidden(path,file_extension):
    return sorted(glob(os.path.join(path, '*.'+file_extension)))

# DATA_DIR = './data_RGB'
DATA_DIR = args.data_dir
# set the result foder and temp folder
result_dir=os.path.join(DATA_DIR,args.sub_dataset+'_'+args.Encoder)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
temp_img_dir=os.path.join(DATA_DIR,'aug_'+args.sub_dataset+'_'+args.Encoder)
if not os.path.exists(temp_img_dir):
    os.makedirs(temp_img_dir)

# print(DATA_DIR)
model_save_dir=os.path.join(result_dir,'model_save')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
# print(os.path.join(DATA_DIR,'images',args.sub_dataset))
print(os.path.join(DATA_DIR,'images',args.sub_dataset))
images_dir=listdir_nohidden(os.path.join(DATA_DIR,args.sub_dataset,'images'),'jpg')
# print(images_dir)
single_folder=True
if os.path.isdir(images_dir[0]):# then will run 4 folders
    single_folder=False
    # images_dir=os.path.join(DATA_DIR,'images',args.sub_dataset)
# print(os.path.join(DATA_DIR,'images',args.sub_dataset))
labels_dir=listdir_nohidden(os.path.join(DATA_DIR,args.sub_dataset,'labels'),'png')
# if  os.path.isdir(labels_dir[0]):
    # labels_dir=os.path.join(DATA_DIR,'labels',args.sub_dataset)
# Batch_size=100
Batch_size=args.Batch_size
# Set num of epochs
# EPOCHS = 50    
EPOCHS = args.EPOCHS  
Train_with_gray=args.Force_to_Gray
# Train_with_gray=True

# =================================================================================================   
# flag_dir=os.path.join(DATA_DIR, 'temp_train')
 # copy file from image and label folder to other traget folders
# def img_label_copy(root,source, target_dir):
#     for filename in source:
#         if filename.endswith(".jpg") or filename.endswith(".png") :
#             # fileparts=filename.split('\\')            
#             if not os.path.isdir(os.path.join(root, target_dir)):
#                 os.mkdir(os.path.join(root, target_dir))
#             copyfile(filename,os.path.join(root, target_dir,ntpath.basename(filename)))
#             # copyfile(filename,os.path.join(root, target_dir,fileparts[-1]))
def img_label_copy(source, target_dir):
    for filename in source:
        if filename.endswith(".jpg") or filename.endswith(".png") :
            # fileparts=filename.split('\\')            
            if not os.path.isdir( target_dir):
                os.mkdir( target_dir)
            copyfile(filename,os.path.join(target_dir,ntpath.basename(filename)))
# image augmentation for nine times 
img_transform = album.Compose ([
    album.RandomCrop(height=512, width=512, always_apply=True),# make sure 512*512
    album.HorizontalFlip(always_apply=True,p=1),
    album.VerticalFlip(always_apply=True,p=1),
    album.RandomRotate90(always_apply=True,p=1),
    album.Transpose(always_apply=True,p=1),
    album.AdvancedBlur(blur_limit=(3,7),sigmaX_limit=(0.2, 1.0), 
                       sigmaY_limit=(0.2, 1.0), rotate_limit=90, 
                       beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), 
                       always_apply=True,p=1),# default parmaters
    album.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True,p=1),
    album.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=True,p=1),
    album.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True,p=1)    
    ])

# train_img_dir=os.path.join(DATA_DIR, 'aug/train/img/')
# train_label_dir=os.path.join(DATA_DIR, 'aug/train/label/')
# valid_img_dir=os.path.join(DATA_DIR, 'aug/valid/img/')
# valid_label_dir=os.path.join(DATA_DIR, 'aug/valid/label/')

# train_img_dir=os.path.join(DATA_DIR, 'aug_'+args.sub_dataset,'aug1','train','img')
# train_label_dir=os.path.join(DATA_DIR, 'aug_'+args.sub_dataset,'aug1','train','label')
# valid_img_dir=os.path.join(DATA_DIR, 'aug_'+args.sub_dataset,'aug1','valid','img')
# valid_label_dir=os.path.join(DATA_DIR, 'aug_'+args.sub_dataset,'aug1','valid','label')
test_img_dir=os.path.join(temp_img_dir,'test','img')
test_label_dir=os.path.join(temp_img_dir,'test','label')
if not os.path.exists(test_img_dir):
    os.makedirs(test_img_dir)
    os.makedirs(test_label_dir)
else:
    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_label_dir)
    os.makedirs(test_img_dir)
    os.makedirs(test_label_dir)

    


''' 
#this part is used to split images and labels to target folders, don't want to use since will taka a lot of space to storge
flag_dir=os.path.join(DATA_DIR, 'test')

if not os.path.isdir(flag_dir):
    x_train_dir,x_test_dir,y_train_dir,y_test_dir=train_test_split(path_img,path_label,test_size=0.6)
    x_test_dir,x_valid_dir,y_test_dir,y_valid_dir=train_test_split(x_test_dir,y_test_dir,test_size=0.5)
    img_label_copy(DATA_DIR,x_train_dir,'train')
    img_label_copy(DATA_DIR,y_train_dir,'train_labels')
    img_label_copy(DATA_DIR,x_test_dir,'test')
    img_label_copy(DATA_DIR,y_test_dir,'test_labels')
    img_label_copy(DATA_DIR,x_valid_dir,'val')
    img_label_copy(DATA_DIR,y_valid_dir,'val_labels')


x_train_dir = os.path.join(DATA_DIR, 'train')# string
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')
'''

class_names =['background','rivers']
# Get class RGB values
class_rgb_values =[[0,0,0],[255,255,255]]

print('All dataset classes and their corresponding RGB values in labels:')

#RGB:white-255,255,255; black:0,0,0
select_classes = ['background', 'rivers']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# define the dataset building class

class BuildingsDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)


# dataset = BuildingsDataset(Xtrain, Ytrain, class_rgb_values=select_class_rgb_values)
# dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)

# random_idx = random.randint(0, len(dataset)-1)
# image, mask = dataset[2]

# visualize(
#     original_image = image,
#     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     one_hot_encoded_mask = reverse_one_hot(mask)
# )

def get_training_augmentation():
    train_transform = [
        # album.RandomCrop(height=256, width=256, always_apply=True),
        # album.OneOf(
        #     [
        #         album.HorizontalFlip(p=1),
        #         album.VerticalFlip(p=1),
        #         album.RandomRotate90(p=1),
        #     ],
        #     p=0.75,
        # ),
        album.RandomCrop(height=512, width=512, always_apply=True),# make sure 512*512
        # album.HorizontalFlip(p=1),
        # album.VerticalFlip(p=1),
        # album.RandomRotate90(p=1),
        # album.Transpose(p=1),
        # album.AdvancedBlur(blur_limit=(3,7),sigmaX_limit=(0.2, 1.0), 
        #                    sigmaY_limit=(0.2, 1.0), rotate_limit=90, 
        #                    beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), 
        #                    always_apply=True),# default parmaters
        # album.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True),
        # album.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=True),
        # album.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True)        
        
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0)
        # album.RandomCrop(height=512, width=512, always_apply=True),# make sure 512*512
        # album.HorizontalFlip(p=1),
        # album.VerticalFlip(p=1),
        # album.RandomRotate90(p=1),
        # album.Transpose(p=1),
        # album.AdvancedBlur(blur_limit=(3,7),sigmaX_limit=(0.2, 1.0), 
        #                    sigmaY_limit=(0.2, 1.0), rotate_limit=90, 
        #                    beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), 
        #                    always_apply=True),# default parmaters
        # album.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True),
        # album.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=True),
        # album.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True)
        
        
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)

'''

augmented_dataset = BuildingsDataset(
    x_train_dir, y_train_dir,
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset)-1)

#Different augmentations on a random image/mask pair (256*256 crop)
for i in range(3):
    image, mask = augmented_dataset[random_idx]
    visualize(
        original_image = image,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask)
    )
'''

########################################################################################################################
# ENCODER = 'resnet152' #batchsize=2
ENCODER_WEIGHTS = 'imagenet'
# ENCODER = 'resnet50' #batchsize=2
ENCODER=args.Encoder
# ENCODER = 'resent152' #batchsize=16 
# ENCODER_WEIGHTS = 'imagenet'

CLASSES = class_names
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

## load pretrained model





preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
#if torch.cuda.device_count()>1:
   # print("Let's use ",torch.cuda.device_count(),"GPUs!")
  #  model=nn.DataParallel(model)

TRAINING = True

# Set num of epochs
# EPOCHS = 50

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for cuda
print('Use '+str(torch.cuda.device_count())+' GPUs!')
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")# for M1
# define loss function
loss = smp.utils.losses.DiceLoss()
if torch.cuda.device_count()>1:
    print("Let's use ",torch.cuda.device_count(),"GPUs!")
    model=nn.DataParallel(model)
model.to(DEVICE)
# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.3),
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.Recall(),
]

# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
# define learning rate scheduler (not used in this NB)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=1, T_mult=2, eta_min=5e-5,
# )

# load best saved model checkpoint from previous commit (if present)
# if os.path.exists(r'E:\Wenqi\DeeplabV3p_kaggle\all_resnet152\model_save/best_model.pt'):
#     model.load_state_dict(r'E:\Wenqi\DeeplabV3p_kaggle\all_resnet152\model_save/best_model.pt')
#     # model = torch.load(r'E:\Wenqi\DeeplabV3p_kaggle\all_resnet152\model_save/best_model.pth', map_location=DEVICE)

# change the model load method since the model trained by multi gpu
stat_dict_path='./best_model.pt'
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(stat_dict_path,map_location=DEVICE).items()})
# model.load_state_dict(torch.load(stat_dict_path,map_location=DEVICE))
# optimizer.load_state_dict(model['optimizer_state_dict'])
# loss=model['loss']
# epoch=model['epoch']
model.eval()
model.to(DEVICE)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
########################################################################################################################

# for cross-validation
path_img=[]
path_label=[]
class_flag=[]
j=1
# print(images_dir)
if single_folder:
    # temp=sorted(glob(os.path.join(images_dir, '*.jpg')))    
    # path_img.extend(temp)    
    # class_flag.extend([j]*len(temp))
    # path_label.extend(sorted(glob(os.path.join(labels_dir, '*.png'))))
       
    path_img=images_dir    
    class_flag=[j]*len(path_img)
    path_label=labels_dir
    # path_label.extend(sorted(glob(os.path.join(labels_dir, '*.png'))))
else:
    for i in images_dir:        
        temp=sorted(glob(os.path.join(i, '*.jpg')))    
        path_img.extend(temp)    
        class_flag.extend([j]*len(temp))
        j=j+1
    for l in labels_dir:    
        path_label.extend(sorted(glob(os.path.join(l, '*.png'))))
# print('path_img: '+str(path_img))
# write path_img and path_label
# path_img.sort()
# path_label.sort()
with open(os.path.join(result_dir,'path_img.txt'), 'w') as fp:
    for item in path_img:
        # write each item on a new line
        fp.write("%s\n" % item)
with open(os.path.join(result_dir,'path_label.txt'), 'w') as fp:
    for item in path_label:
        # write each item on a new line
        fp.write("%s\n" % item)
path_img=np.array(path_img)
path_label=np.array(path_label)

# here, divide the dataset to train/valid and test subdataset, x_train_dir actually include the train and valid subdataset
x_train_dir,x_test_dir,y_train_dir,y_test_dir=train_test_split(path_img,path_label,test_size=0.2)
img_label_copy(x_test_dir, test_img_dir)
img_label_copy(y_test_dir, test_label_dir)
path_img=x_train_dir
path_label=y_train_dir
j=1
if single_folder:       
    class_flag=[j]*len(path_img)
else:
    for i in images_dir:        
        temp=sorted(glob(os.path.join(i, '*.jpg')))    
        path_img.extend(temp)    
        class_flag.extend([j]*len(temp))
        j=j+1
    for l in labels_dir:    
        path_label.extend(sorted(glob(os.path.join(l, '*.png'))))



skf = StratifiedKFold(n_splits=args.FoldNumber)
start_time=time.time()
def cv_train(Xtrain, Xtest,Ytrain, Ytest):
    train_img_dir=os.path.join(temp_img_dir,'aug'+str(fold),'train','img')    
    train_label_dir=os.path.join(temp_img_dir,'aug'+str(fold),'train','label')
    valid_img_dir=os.path.join(temp_img_dir,'aug'+str(fold),'valid','img')
    valid_label_dir=os.path.join(temp_img_dir,'aug'+str(fold),'valid','label')
    
    # train_img_dir=os.path.join(DATA_DIR, 'aug'+str(fold)+'/train/img/')
    # train_label_dir=os.path.join(DATA_DIR, 'aug'+str(fold)+'/train/label/')
    # valid_img_dir=os.path.join(DATA_DIR, 'aug'+str(fold)+'/valid/img/')
    # valid_label_dir=os.path.join(DATA_DIR, 'aug'+str(fold)+'/valid/label/')
    folder_flag=0
    if not os.path.isdir(os.path.join(temp_img_dir, 'aug'+str(fold))):
        if fold>=2:
            shutil.rmtree(os.path.join(temp_img_dir, 'aug'+str(fold-1)))  
        os.makedirs(train_img_dir)
        os.makedirs(train_label_dir)
        os.makedirs(valid_img_dir)
        os.makedirs(valid_label_dir)
        
    elif os.path.isdir(os.path.join(temp_img_dir,'aug'+str(fold))):
            folder_flag=1
    # if not os.path.isdir(os.path.join(DATA_DIR, 'aug')):
    #     os.makedirs(train_img_dir)
    #     os.makedirs(train_label_dir)
    #     os.makedirs(valid_img_dir)
    #     os.makedirs(valid_label_dir)
    # elif os.path.isdir(os.path.join(DATA_DIR, 'aug')):
    #         # os.rmdir(os.path.join(DATA_DIR, 'aug'))
    #         shutil.rmtree(os.path.join(DATA_DIR, 'aug'))
    #         os.makedirs(train_img_dir)
    #         os.makedirs(train_label_dir)
    #         os.makedirs(valid_img_dir)
    #         os.makedirs(valid_label_dir)
    stime=time.time()
    if folder_flag==0: 
        index=1    
        for img_temp,mask_temp in zip(Xtrain,Ytrain):    
            for aug in img_transform:
                # here change the input image to gray 
                image_to_read=cv2.imread(img_temp)
                if Train_with_gray: # if choose to train with gray imgs
                    if not ((image_to_read[:,:,0]==image_to_read[:,:,1]).all() and (image_to_read[:,:,1]==image_to_read[:,:,2]).all()):# if the input img is RGB, then convert to gray
                    
                        gray = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)
                        img2 = np.zeros_like(image_to_read)
                        img2[:,:,0] = gray
                        img2[:,:,1] = gray
                        img2[:,:,2] = gray
                        image_to_read=img2
                augmented=aug(image=image_to_read,mask=cv2.imread(mask_temp))       
                #augmented=aug(image=cv2.imread(img_temp),mask=cv2.imread(mask_temp))        
                # aug_img=aug(image=cv2.imread(img_temp))['image']        
                cv2.imwrite(os.path.join(train_img_dir, 'img_'+str(index))+'.jpg',augmented['image'])
                cv2.imwrite(os.path.join(train_label_dir, 'label_'+str(index))+'.png',augmented['mask'])
                # visualize(augmented['image'], augmented['mask'])
                index=index+1
        index=1    
        for img_temp,mask_temp in zip(Xtest,Ytest):    
            for aug in img_transform:
                # here change the input image to gray 
                image_to_read=cv2.imread(img_temp)
                if Train_with_gray: # if choose to train with gray imgs
                    if not ((image_to_read[:,:,0]==image_to_read[:,:,1]).all() and (image_to_read[:,:,1]==image_to_read[:,:,2]).all()):# if the input img is RGB, then convert to gray
                    
                        gray = cv2.cvtColor(image_to_read, cv2.COLOR_BGR2GRAY)
                        img2 = np.zeros_like(image_to_read)
                        img2[:,:,0] = gray
                        img2[:,:,1] = gray
                        img2[:,:,2] = gray
                        image_to_read=img2
                augmented=aug(image=image_to_read,mask=cv2.imread(mask_temp)) 
                # augmented=aug(image=cv2.imread(img_temp),mask=cv2.imread(mask_temp))
                cv2.imwrite(os.path.join(valid_img_dir, 'img_'+str(index))+'.jpg',augmented['image'])
                cv2.imwrite(os.path.join(valid_label_dir, 'label_'+str(index))+'.png',augmented['mask'])
                index=index+1
    etime=time.time()
    print('Time to augment images: '+str(round(etime-stime,2))) 
    x_train_dir=train_img_dir
    y_train_dir=train_label_dir        
    x_valid_dir = valid_img_dir
    y_valid_dir = valid_label_dir
    
    train_dataset = BuildingsDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    
    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    
    # Get train and val data loaders
    #禁用多线程
    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, pin_memory=True,num_workers=0,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False, pin_memory=True,num_workers=0,drop_last=True)
    # train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, pin_memory=True,num_workers=torch.cuda.device_count()*6,drop_last=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False, pin_memory=True,num_workers=torch.cuda.device_count()*6,drop_last=True)
    ########################################################################################################################
    # if TRAINING:
    
    best_iou_score = 0.0
    output_iou_score=0.0
    train_logs_list, valid_logs_list = [], []
    model.train()
    for i in range(0, EPOCHS):
    
        # Perform training & validation
        print('\n Current Fold: {}, Epoch: {}'.format(fold,i))
        train_logs = train_epoch.run(train_loader)
       # print("4-torch.cuda.memory_allocated: \n")
       # gpu_usage()
        valid_logs = valid_epoch.run(valid_loader)
        #print("5-torch.cuda.memory_allocated: \n")
        #gpu_usage()        
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
    
        # Save model if a better val IoU score is obtained
        # if best_iou_score < valid_logs['iou_score']:
        #     best_iou_score = valid_logs['iou_score']
        #     torch.save(model, './data/best_model.pth')
        #     print('Model saved!')
        if best_iou_score < valid_logs['fscore']:
            best_iou_score = valid_logs['fscore']
            torch.save(model,model_save_dir+ '/best_model.pth')
            torch.save(model.state_dict(), model_save_dir+ '/best_model.pt')
            print('Best Model saved!')
        # if valid_logs['fscore'] >= output_iou_score +0.02:
        #     torch.save(model,model_save_dir+'/best_model_'+str(round(best_iou_score,2))+'_'+str(fold)+'_'+str(i)+'.pth')
        #     torch.save(model.state_dict(),model_save_dir+'/best_model_'+str(round(best_iou_score,2))+'_'+str(fold)+'_'+str(i)+'_.pt')
        #     print('Model_'+str(round(best_iou_score,2))+'_'+str(fold)+'_'+str(i)+'_saved!!!')
        #     output_iou_score=valid_logs['fscore']

    
    return train_logs_list,valid_logs_list

if not os.path.exists(os.path.join(model_save_dir,'valid_logs_list.txt')):
    ff=open(os.path.join(model_save_dir,'valid_logs_list.txt'),'x')
fold=1
valid_avg_list=[]

for train, test in skf.split(list(range(len(path_img))),class_flag):
    print('Current Fold: '+str(fold))
    print("%s %s" % (train, test))
    Xtrain, Xtest=path_img[train], path_img[test]
    Ytrain, Ytest=path_label[train], path_label[test]
    s_time=time.time()
    train_logs_list,valid_logs_list=cv_train(Xtrain, Xtest,Ytrain, Ytest)
    valid_avg_list.append(valid_logs_list)
    dt_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    e_time=time.time()
    m,s=divmod(e_time-s_time,60)
    h,m=divmod(m,60)
    train_time_fold=str(int(h))+':'+str(int(m))+':'+str(int(s))
    ff=open(os.path.join(model_save_dir,'valid_logs_list.txt'),'a')
    ff.write(dt_time+', Fold: '+str(fold)+', '+'Time: '+train_time_fold+', '+str(valid_logs_list)+'\n')
    ff.close()       
    fold=fold+1
   # train_logs_list,valid_logs_list=cv_train(Xtrain, Xtest,Ytrain, Ytest)
   # ff=open('./valid_logs_list.txt','a')
   # ff.write(log)
   # ff.close()       
            
end_time=time.time()
m,s=divmod(end_time-start_time,60)
h,m=divmod(m,60)
train_time=str(int(h))+':'+str(int(m))+':'+str(int(s))
print('Train time: ',train_time)

# reorginze the valid_avg_list 
k=0
temp_dict={}
for n in valid_avg_list:    
    temp_dict[k]=n[0]
    k=k+1
    
# mean_dict = {}
# for key in temp_dict[0].keys():
#     mean_dict[key] = sum(temp_dict[n][key] for n in range(len(temp_dict)))/ len(temp_dict)
    # sum(d[key] for d in temp_dict) / len(temp_dict)
    
def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = round(sum(dict_list[d][key] for d in range(len(dict_list)))/ len(dict_list),4)
    return mean_dict
valid_avg_result=dict_mean(temp_dict)

# write the log file
dt_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#train_time=time.strftime("%H:%M:%S",time.gmtime((end_time-start_time)))
# log='['+dt_time+', Train_with_gray: {}'.format(Train_with_gray)+', Dataset: {}'.format(args.sub_dataset)+', Encoder: {}'.format(ENCODER)+', Epoch: {}'.format(EPOCHS)+', Batch Size: {}'.format(Batch_size)+', Train Time: '+train_time+', Performace: '+str(valid_avg_result)+']'+'\n'
log='['+dt_time+', Computer: {}'+', Train_with_gray: {}'+', Dataset: {}'+', Encoder: {}'+'， K-Fold: {}'+', Epoch: {}'+', Batch Size: {}'+', Train Time: {}'+', Performace: {}'+']'+'\n'


if not os.path.exists(os.path.join(model_save_dir,'cv_train_log.txt')):
    f=open(os.path.join(model_save_dir,'cv_train_log.txt'),'x')

f=open(os.path.join(model_save_dir,'cv_train_log.txt'),'a')
# f.write(log)
f.write(log.format(socket.gethostname(),Train_with_gray,args.sub_dataset,ENCODER,args.FoldNumber,EPOCHS,Batch_size,train_time,valid_avg_result))
f.close()



'''
Load the best model
'''
# best_model= smp.DeepLabV3Plus(
#     encoder_name=ENCODER,
#     encoder_weights=ENCODER_WEIGHTS,
#     classes=len(CLASSES),
#     activation=ACTIVATION,
# )
# if os.path.exists(model_save_dir+ '/best_model.pt'):    
#     # best_model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(stat_dict_path,map_location=DEVICE).items()})
#     best_model.load_state_dict(torch.load(model_save_dir+ '/best_model.pt',map_location=DEVICE))
#     best_model.eval()
if os.path.exists(os.path.join(model_save_dir, 'best_model.pth')):
    best_model = torch.load(os.path.join(model_save_dir, 'best_model.pth'), map_location=DEVICE)
    print('Loaded DeepLabV3+ model!!!')
# # # load best saved model checkpoint from previous commit (if present)
# # elif os.path.exists('./input//deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth'):
# #     best_model = torch.load('./input//deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth', map_location=DEVICE)
# #     print('Loaded DeepLabV3+ model from a previous commit.')
# # create test dataloader (with preprocessing operation: to_tensor(...))
test_dataset = BuildingsDataset(
    test_img_dir,
    test_label_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

# test_dataloader = DataLoader(test_dataset,batch_size=Batch_size, shuffle=True, pin_memory=True,num_workers=torch.cuda.device_count()*6,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=Batch_size, shuffle=True, pin_memory=True,num_workers=0,drop_last=True)

# # test dataset for visualization (without preprocessing transformations)
test_dataset_vis = BuildingsDataset(
    test_img_dir,
    test_label_dir,
    augmentation=get_validation_augmentation(),
    class_rgb_values=select_class_rgb_values,
)
# get a random test image/mask index
random_idx = random.randint(0, len(test_dataset_vis)-1)
image, mask = test_dataset_vis[random_idx]

# visualize(
#     original_image = image,
#     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     one_hot_encoded_mask = reverse_one_hot(mask)
# )
# # Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[512, 512, 3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
            padding:image_size - padding,
            padding:image_size - padding,
            :,
            ]
sample_preds_folder = os.path.join(DATA_DIR,args.sub_dataset+'_'+args.Encoder,'sample_preds_folder')
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)

for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    # Convert pred_mask from `CHW` format to `HWC` format
    pred_mask = np.transpose(pred_mask,(1,2,0))
    # Get prediction channel corresponding to building
    pred_river_heatmap = pred_mask[:,:,select_classes.index('rivers')]
    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))
    # Convert gt_mask from `CHW` format to `HWC` format
    gt_mask = np.transpose(gt_mask,(1,2,0))
    gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
    
    #just comment here for test faster ##########################################################################################################
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])

    # visualize(
    #     original_image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=pred_mask,
    #     predicted_river_heatmap=pred_river_heatmap
    # )
test_epoch = smp.utils.train.ValidEpoch(
model,
loss=loss, 
metrics=metrics, 
device=DEVICE,
verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
print(f"Mean Accuracy: {valid_logs['accuracy']:.4f}")
print(f"Mean F_score: {valid_logs['fscore']:.4f}")
print(f"Mean Recall: {valid_logs['recall']:.4f}")
train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df.T
plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title(args.sub_dataset+'_IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(result_dir,args.sub_dataset+'_iou_score_plot.png'))
# plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Dice Loss', fontsize=20)
plt.title(args.sub_dataset+'_Dice Loss Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(result_dir,args.sub_dataset+'_dice_loss_plot.png'))
# plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.accuracy.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.accuracy.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title(args.sub_dataset+'_Accuracy Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(result_dir,args.sub_dataset+'_accuracy_plot.png'))

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.fscore.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.fscore.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Fscore', fontsize=20)
plt.title(args.sub_dataset+'_Fscore Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(result_dir,args.sub_dataset+'_fscore_plot.png'))

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.recall.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.recall.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Recall', fontsize=20)
plt.title(args.sub_dataset+'_Recall Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(result_dir,args.sub_dataset+'_recall_plot.png'))

# this part is used to send email after trainning finished
import smtplib
from email.mime.text import MIMEText

smtp_ssl_host = 'smtp.163.com'  # smtp.mail.yahoo.com
smtp_ssl_port = 465
username = 'zzczwxy_scu@163.com'
password = '930913LWQ'
sender = username
targets = 'zzczwxy@hotmail.com'

# (log.format(socket.gethostname(),Train_with_gray,args.sub_dataset,ENCODER,args.FoldNumber,EPOCHS,Batch_size,train_time,valid_avg_result))
msg = MIMEText(log.format(socket.gethostname(),Train_with_gray,args.sub_dataset,ENCODER,args.FoldNumber,EPOCHS,Batch_size,train_time,valid_avg_result))
msg['Subject'] = 'SCU WorkStation:Information of CNN Model PostTraining'
msg['From'] = sender
msg['To'] = targets #', '.join(targets)

server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
server.login(username, password)
server.sendmail(sender, targets, msg.as_string())
server.quit()    