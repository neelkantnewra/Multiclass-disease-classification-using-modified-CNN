import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from tqdm.notebook import tqdm_notebook
from keras.models import load_model
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam

from model.umodel import dice_coef,dice_coef_loss,unet
from tools.imgsegment import SegmentImage

IMG_SIZE = 512
LR = 1e-5

# U-Net model
model = unet(input_size=(512,512,1))
model.load_weight("weight.best.hdf5")
model.compile(optimizer=Adam(learning_rate=LR), loss=dice_coef_loss, metrics=['binary_accuracy'])

cat_path = '../input/chest-xray-pneumonia/chest_xray/train'
categories = os.listdir(cat_path)

labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories,labels))

directory = ['train']
data_path = "../input/chest-xray-pneumonia/chest_xray"
train_data = []
train_target = []
for path in tqdm_notebook(directory):
    image_dir_path = os.path.join(data_path,path)
    image_dir_file = os.listdir(image_dir_path)
    for image in tqdm_notebook(image_dir_file):
            image_path = os.path.join(image_dir_path,image)
            image_path_file = os.listdir(image_path)
            for img_name in tqdm_notebook(image_path_file):
                img_path = os.path.join(image_path,img_name)
                try:
                    maskapply,masked_image,chest_image = SegmentImage(model=model,path=img_path,img_shape = (IMG_SIZE,IMG_SIZE),threshold = 0.5)
                    train_data.append(masked_image)
                    train_target.append(label_dict[image])
                except Exception as e:
                    print("Exception: ",e)
                    
data = np.array(train_data)/255.0
data = np.reshape(data,(data.shape[0],IMG_SIZE,IMG_SIZE,1))
target = np.array(train_target)

new_target = np_utils.to_categorical(target)
np.save('train_data',data)
np.save('train_target',new_target)                    

directory = ['val']
data_path = "../input/chest-xray-pneumonia/chest_xray"

val_data = []
val_target = []
for path in tqdm_notebook(directory):
    image_dir_path = os.path.join(data_path,path)
    image_dir_file = os.listdir(image_dir_path)
    for image in tqdm_notebook(image_dir_file):
            image_path = os.path.join(image_dir_path,image)
            image_path_file = os.listdir(image_path)
            for img_name in tqdm_notebook(image_path_file):
                img_path = os.path.join(image_path,img_name)
                try:
                    maskapply,masked_image,chest_image = SegmentImage(model=model,path=img_path,img_shape = (IMG_SIZE,IMG_SIZE),threshold = 0.5)
                    val_data.append(masked_image)
                    val_target.append(label_dict[image])
                except Exception as e:
                    print("Exception: ",e)

data = np.array(val_data)/255.0
data = np.reshape(data,(data.shape[0],IMG_SIZE,IMG_SIZE,1))
target = np.array(val_target)

new_target = np_utils.to_categorical(target)
np.save('val_data',data)
np.save('val_target',new_target)

directory = ['test']
data_path = "../input/chest-xray-pneumonia/chest_xray"

test_data = []
test_target = []
for path in tqdm_notebook(directory):
    image_dir_path = os.path.join(data_path,path)
    image_dir_file = os.listdir(image_dir_path)
    for image in tqdm_notebook(image_dir_file):
            image_path = os.path.join(image_dir_path,image)
            image_path_file = os.listdir(image_path)
            for img_name in tqdm_notebook(image_path_file):
                img_path = os.path.join(image_path,img_name)
                try:
                    maskapply,masked_image,chest_image = SegmentImage(model=model,path=img_path,img_shape = (IMG_SIZE,IMG_SIZE),threshold = 0.5)
                    test_data.append(masked_image)
                    test_target.append(label_dict[image])
                except Exception as e:
                    print("Exception: ",e)
               

data = np.array(test_data)/255.0
data = np.reshape(data,(data.shape[0],IMG_SIZE,IMG_SIZE,1))
target = np.array(test_target)

from keras.utils import np_utils

new_target = np_utils.to_categorical(target)
np.save('test_data',data)
np.save('test_target',new_target)

print(f"Total number of normal case: {val_target.count(1)+test_target.count(1)+train_target.count(1)}")
print(f"Total number of pneumonia case: {val_target.count(0)+test_target.count(0)+train_target.count(0)}")

print(f"Total number of normal case test: {test_target.count(1)}")
print(f"Total number of pneumonia case test: {test_target.count(0)}")

print(f"Total number of normal case val: {val_target.count(1)}")
print(f"Total number of pneumonia case val: {val_target.count(0)}")

print(f"Total number of normal case train: {train_target.count(1)}")
print(f"Total number of pneumonia case train: {train_target.count(0)}")
