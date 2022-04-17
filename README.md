# Multiclass disease classification using Modified CNN

# About
We will be modifing and testing various model which have achieved higher accuracy for multiclass classification problem to achieve good result.

# Method

We have collected Chest X-Ray(CXR) images of various dimensions with label `Pneumonia`,`Normal`,`Covid-19`,`Tuberculosis`, and resize it to `128x128` pixel grayscale images. 
We have developed two types of dataset:

- Segmented Image Dataset
- Normal Image Dataset

## Segmented Image Dataset
We have used U-Net model for the prediction of mask for selecting only lung portion. Although Accuracy of U-Net model was itself very low so we get very few clean segmented mask. This may be the reason we are getting low accuracy for the segmented data. 

<img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Image/Segmented-data.png">

### Why we want Mask?
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

# Analysis

## Model Architecture

### VGG-13
<img src="https://github.com/neelkantnewra/Multiclass-disease-classification-using-modified-CNN/blob/main/Image/Model-Architecture/VGG-13/VGG-13.png">

## Model Performance
### Without Segmentation

#### Metric Table
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
      <td align="center">0.9512</td>
      <td align="center">0.9531</td>
      <td align="center">0.9485</td>
      <td align="center">0.9508</td>
      <td align="center">0.8560</td>
      <td align="center">0.8620</td>
      <td align="center">0.8508</td>
      <td align="center">0.8564</td>
    </tr>
    <tr>
      <td>AlexNet</td>
      <td align="center">0.9683</td>
      <td align="center">0.9709</td>
      <td align="center">0.9658</td>
      <td align="center">0.9683</td>
      <td align="center">0.8197</td>
      <td align="center">0.8337</td>
      <td align="center">0.8132</td>
      <td align="center">0.8233</td>
    </tr>
    <tr>
      <td>MobileNet</td>
      <td align="center">0.9918</td>
      <td align="center">0.9924</td>
      <td align="center">0.9915</td>
      <td align="center">0.9919</td>
      <td align="center">0.8457</td>
      <td align="center">0.8449</td>
      <td align="center">0.8405</td>
      <td align="center">0.8427</td>
    </tr>
  </tbody>
</table>

#### Confusion Metrics

<table>
<tbody>
  <tr>
      <td align="center">VGG-13</td>
      <td align="center">AlexNet</td>
      <td align="center">MobileNet</td>
    </tr>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/AlexNet/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/MobileNet/confusion-matrix.png" width="400px"> </td>
  </tr>
  <tr>
      <td align="center">Modified-DarkCovidNet</td>
      <td align="center">AlexNet</td>
      <td align="center">MobileNet</td>
    </tr>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/Modified-DarkCovidNet/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/AlexNet/confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/MobileNet/confusion-matrix.png" width="400px"> </td>
  </tr>
</tbody>
</table>


### With segmentation
#### Metric Table
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
      <td align="center">0.9317</td>
      <td align="center">0.9367</td>
      <td align="center">0.9287</td>
      <td align="center">0.9327</td>
      <td align="center">0.7613</td>
      <td align="center">0.7668</td>
      <td align="center">0.7549</td>
      <td align="center">0.7608</td>
    </tr>
    <tr>
      <td>AlexNet</td>
      <td align="center">0.9625</td>
      <td align="center">0.9652</td>
      <td align="center">0.9594</td>
      <td align="center">0.9623</td>
      <td align="center">0.7691</td>
      <td align="center">0.7781</td>
      <td align="center">0.7639</td>
      <td align="center">0.7709</td>
    </tr>
    <tr>
      <td>MobileNet</td>
      <td align="center">0.9344</td>
      <td align="center">0.9389</td>
      <td align="center">0.9301</td>
      <td align="center">0.9345</td>
      <td align="center">0.7756</td>
      <td align="center">0.7795</td>
      <td align="center">0.7704</td>
      <td align="center">0.7749</td>
    </tr>
  </tbody>
</table>

#### Confusion Metrics

<table>
<tbody>
  <tr>
      <td align="center">VGG-13</td>
      <td align="center">AlexNet</td>
      <td align="center">MobileNet</td>
    </tr>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/VGG13/segmented-confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/AlexNet/segmented-confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/MobileNet/segmented-confusion-matrix.png" width="400px"> </td>
  </tr>
  <tr>
      <td align="center">Modified-DarkCovidNet</td>
      <td align="center">AlexNet</td>
      <td align="center">MobileNet</td>
    </tr>
  <tr>
  <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/Modified-DarkCovidNet/segmented-confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/AlexNet/segmented-confusion-matrix.png" width="400px"> </td>
    <td><img src="https://github.com/neelkantnewra/Multiclass-disease-detection-using-modified-CNN/blob/main/Analysis/MobileNet/segmented-confusion-matrix.png" width="400px"> </td>
  </tr>
</tbody>
</table>
