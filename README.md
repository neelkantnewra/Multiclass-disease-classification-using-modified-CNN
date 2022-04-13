# Multiclass disease detection using Modified CNN

About
-----
We will be modifing and testing various model which have achieved higher accuracy for multiclass classification problem to achieve good result.

Method
------

We have collected Chest X-Ray(CXR) images of various dimensions with label `Pneumonia`,`Normal`,`Covid-19`,`Tuberculosis`, and resize it to `128x128` pixel grayscale images. 
We have developed two types of dataset:

- Segmented Image Dataset
- Normal Image Dataset

Segmented Image Dataset
-----------------------
We have used U-Net model for the prediction of mask for selecting only lung portion. Although Accuracy of U-Net model was itself very low so we get very few clean segmented mask. This may be the reason we are getting low accuracy for the segmented data. 

## Why we want Mask?
Mask will assure us that our model is learning on the right data of the CXR, and build the confidence among user, It can also be helpful for skipping our model at certain pixel of the images.

## Normal Image Dataset

It is the simple CXR images which is flatten to form ndarray.

target are kept categorical
| |Total Images|
| :---: | :---: |
|Training| 6326|
|Validating| 38 |
|Testing| 771|

Prepared Dataset Link: https://www.kaggle.com/datasets/newra008/cxr-data-for-multiclass-classification
