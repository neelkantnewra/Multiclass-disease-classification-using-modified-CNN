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
Mask will assure us that our model is learning on the right data of the CXR, and build the confidence among user, It can also be helpful for skipping model computation at certain pixel of the images.

## Normal Image Dataset

It is the simple CXR images which is flatten to form ndarray.

target are kept categorical
| |Total Images|
| :---: | :---: |
|Training| 6326|
|Validating| 38 |
|Testing| 771|

Prepared Dataset Link: [CXR DATA for Multiclass Classfification](https://www.kaggle.com/datasets/newra008/cxr-data-for-multiclass-classification)

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
      <td align="center">0.9783</td>
      <td align="center">0.9788</td>
      <td align="center">0.9780</td>
      <td align="center">0.9783</td>
      <td align="center">0.7743</td>
      <td align="center">0.7775</td>
      <td align="center">0.7704</td>
      <td align="center">0.7739</td>
    </tr>
  </tbody>
</table>

### Confusion Metrics

<table>
<tbody>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/Confusion%20matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/Confusion%20matrix.png" width="400px"> </td>
  </tr>
</tbody>
</table>
