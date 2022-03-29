import os
from os import mkdir

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.keras.optimizers import Adam


def SegmentImage(model,path,img_shape = (512,512),threshold = 0.5):
    '''
    Parameters
    ----------
    model : Segmented model(h5)
    path(string) : file path to image
    img_shape(tuple): shape of the image(IMG_WIDTH,IMG_HEIGHT) used in segmenation model
    threshold : float value varing between 0 and 1, thresholding the mask
    
    Return
    ------
    Segmented mask , segmented image, original image
    '''
    IMG_WIDTH,IMG_HEIGHT = img_shape
    ori_x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ori_x = cv2.resize(ori_x, (IMG_HEIGHT,IMG_WIDTH))
    x = ori_x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)[0] > threshold
    y_pred = y_pred.astype(np.int32)
    plt.imsave('mask.jpeg',np.squeeze(y_pred),cmap='gray')
    maskapply = cv2.imread('mask.jpeg')
    maskapply = cv2.cvtColor(maskapply, cv2.COLOR_BGR2GRAY)
    chest_image = ori_x
    chest_image = cv2.resize(chest_image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_NEAREST)
    masked_image = cv2.bitwise_and(maskapply,chest_image)
    masked_image = cv2.resize(masked_image,(128,128),interpolation=cv2.INTER_NEAREST)
    return maskapply,masked_image,chest_image
  
  
def KMeanSegment(img_path:str,filename:str,K=2,attempt=4)->str:
    '''
    Parameter
    ---------
    img_path : path of the image to be segmented
    filename : folder name to save alll file
    K : number of region K-mean (default value = 2)
    attempt: number of attempt you want to carry on (default value = 4)
    
    Return
    ------
    String with file name in which all data saved,
    Folder consisting of mask, segment region, original image, complete segmented image
    '''
    img = cv2.imread(img_path)
    resize_image = img.reshape((-1,3))
    img_float = np.float32(resize_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,10,1.0)
    ret,label,center = cv2.kmeans(img_float,K,attempt,criteria,10,cv2.KMEANS_PP_CENTERS)
    center=np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    # Creating directory
    mkdir(filename)
    mkdir(filename+"/mask")
    mkdir(filename+"/segment region")
    mkdir(filename+"/original")
    mkdir(filename+"/complete")
    
    cv2.imwrite(filename+"/complete"+"/Complete Segment.jpeg",res2)
    cv2.imwrite(filename+"/original"+"/original.jpeg",img)
    filter = [center[x][0] for x in range(K)]
    filter.sort()
    
    for seg in range(K):
        threshold , image = cv2.threshold(res2,filter[seg],255,cv2.THRESH_BINARY_INV)
        cv2.imwrite(filename+"/mask"+"/mask "+str(seg+1)+".jpeg",image)
        threshold , image = cv2.threshold(res2,filter[seg],filter[seg],cv2.THRESH_BINARY)
        cv2.imwrite(filename+"/segment region"+"/Segment region "+str(seg+1)+".jpeg",image)
        
    return "Segment image is in {}".format(filename)
