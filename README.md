# Multiclass disease classification using Modified CNN

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

<img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Image/Segmented-data.png">

## Why we want Mask?
Mask will assure us that our model is learning on the right data of the CXR, and build the confidence among user, It can also be helpful for skipping model computation at certain pixel of the images.

## Normal Image Dataset

It is the simple CXR images which is flatten to form ndarray.

<img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Image/normal-data.png">

target are kept categorical
| |Total Images|
| :---: | :---: |
|Training| 6326|
|Validating| 38 |
|Testing| 771|

Prepared Dataset Link: [CXR DATA for Multiclass Classfification](https://www.kaggle.com/datasets/newra008/cxr-data-for-multiclass-classification)

## Model Architecture

## Model Performance

### Without Segmentation


<table>
  <tbody>
    <tr>
      <th rowspan="2">Model</th>
      <th align="center" colspan="4">Training</th>
      <th align="center" colspan="4">Testing</th>
    </tr>
    <tr>
      <td align="center">Accuracy</td>
      <td align="center">Precision</td>
      <td align="center">Recall</td>
      <td align="center">F1-score</td>
      <td align="center">Accuracy</td>
      <td align="center">Precision</td>
      <td align="center">Recall</td>
      <td align="center">F1-score</td>
    </tr>
    <tr>
      <td>VGG-13</td>
      <td align="center">0.9831</td>
      <td align="center">0.9835</td>
      <td align="center">0.9829</td>
      <td align="center">0.9831</td>
      <td align="center">0.8184</td>
      <td align="center">0.8179</td>
      <td align="center">0.8158</td>
      <td align="center">0.8168</td>
    </tr>
    <tr>
      <td>AlexNet</td>
      <td align="center">0.9876</td>
      <td align="center">0.9876</td>
      <td align="center">0.9868</td>
      <td align="center">0.9871</td>
      <td align="center">0.7937</td>
      <td align="center">0.8015</td>
      <td align="center">0.7859</td>
      <td align="center">0.7936</td>
    </tr>
  </tbody>
</table>

### Confusion Metrics

<table>
<tbody>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/AlexNet/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/confusion-matrix.png" width="400px"> </td>
  </tr>
</tbody>
</table>
