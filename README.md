# Opacity Detection in The Diagnosis of Pneumonia 

## Introduction 

### 1. What is Pneumonia?

In 2019, a new coronavirus (COVID-19) has infected millions of people around the world and this infection can lead to a health complication known as pneumonia. Pneumonia is a lung inflammation that is caused by viral or bacterial infection. This inflammation fills the tiny air sacs (alveoli) in your lung with fluid and pus that prevents the transfer of oxygen into the bloodstream. This condition subsequently leads to severe symptoms like shortness of breath or hypoxia.

### 2. Why chest X-ray?
Pneumonia can be diagnosed by performing medical lab tests or imaging. A chest X-ray is one of the most reliable imaging methods used in conjuction with other tests to form a full clinical assessment. It is not only used to confirm the infection but also determine the location and extent of the infection. In the setting of the Covid-19 pandemic, chest X-rays play an important role in the early detection of the virus as well as identifying patients at high risk of developing more severe illness. Chest X-rays can provide fast results and are easy to access in most countries where testing kits can be scarce or lab test results are significantly delayed.

### Objective

The goal of this study is to **create a model that accurately detects pulmonary opacity in chest X-rays to assist in the diagnosis of pneumonia.** 


## Data Overview
* Source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
* The dataset contains of 5863 anterior-posterior chest X-ray images and is composed of 2 classes: healthy lung and pneumonia lung. 

![Data](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/pneumonia-normal-example.png)


## Exploratory Data Analsysis (EDA)

The dataset is organized into 3 groups: the training set, validation set, and testing set.
![Original data](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/original-data.png)


There are two problems here: 
1. Validation set only has 16 X-ray samples.  
2. Imbalance classes: 1513 normal cases vs 3980 pneumonia cases

To address these issues, I re-sampled the training and validation sets by splitting them into a 90:10 ratio. For the imbalanced datasets, I oversampled the minority group. Below is the result of the re-sampling dataset. 

![Visualization of original dataset and oversampled dataset](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/oversampled-data.png)

## Image Pre-processing

To further address the imbalance issue, I performed image augmentation to generate more data for the model to train on. With this, I introduced additional variation into the dataset and improve generalizability. 

Below are some examples of an X-ray being slightly rotated, shifted vertically and horizontally, zoomed, and cropped. 
![image-augmentation](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/augmentation-example.png)

## Modeling 

For this project, I created a model from scratch and compared it with another model that uses transfer learning. 

**Key Parameters** 

* Loss function - binary_crossentropy
* Optimizer: Adam with a learning rate of 0.001 (learning rate will be reduced if val_loss is not improved over 3 epochs)
* Metric: Recall
* Epochs: 15
* Batch size: 64

**Table 1. Parameters of the Convolutional Blocks**

 Layer | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
-------|----|----|----|----|----|----|----|----|
Type| SeparableConv| MaxPool |SeparableConv  | MaxPool | SeparableConv | MaxPool |SeparableConv|MaxPool |
Filter | 16 | - | 32 | - | 64| - |  128 | - | 256 |
Kernel size | 3 x 3 | - | 3 x 3 | - |3 x 3 | - |3 x 3 | - |
Padding | same | - | same | - | same | - | same | - |
Activation funcion | Relu | - | Relu | -| Relu | -| Relu | - |
Pooling size | - | 2 x 2| - | 2 x 2| - | 2 x 2| - | 2 x 2

Dropout layers, L2 regularization and EarlyStopping were added to minimize the effect of overfitting. 


**Table 2. Parameters of the Fully connected layers**
Hidden Layer | FC1 | FC2 | FC3
---|---|----| ----|
Units | 512 | 64 | 1 |
Activation function | Relu | Relu | Sigmoid |

## Training and Validating 


![loss-accuracy](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/good-acc-loss.jpeg)


## Results & Intepretation

To determine the optimal threshold for my model, I plotted an ROC curve with an area under curve (AUC) of 0.963. 
![ROC-base](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/bad-roc.png)

![Confusion Matrix Base Model](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/cm-15epoch-transfer.png)

Looking at the confusion matrix, the model is better at detecting pneumonia cases than normal cases. There are a high number of normal cases being misclassfied (False Positives). In a medical setting, False Positives typically only lead to additional testing for confirmation but a False Negative could lead to a doctor overlooking something dangerous or even deadly. This could have a detrimental impact on the patient health. Therefore, I selected Recall as the metric to evaluate my model performance. The recall value refers to the proportion of positive cases identified out of total positive cases. 

I compared the results of my model with the model using DenseNet. After adjusting the threshold, the transfer learning model seems to do worse at capturing the positive cases even though the False Positive cases is slightly reduced. 

![Confusion-matrix-tf](https://github.com/chloengnguyen/opacity-detection-chest-xray/blob/master/graph/confusion-matrix-tf-avg.png)

I decided to dig a little deeper and looked at some of my model's predictions. The second X-ray on the first row is being misclassified as a Pneumonia case with a rather high probability. By comparing it with the other normal cases, the X-ray image looks a bit more fuzzy than normal. This could be a result of under-exposure which lowers the quality of the image, and my model is not able to recognize this. It's also possible that my model has a very narrow view of what a normal chest X-ray looks like as the normal class is underrepresented. 


## Conclusion
The model can be improved with more hyperparamenter tuning and further feature selection.

Future improvement: 
    * Advanced data augmentation
    * Class activation mapping (CMAP)
    


## Citations
1. G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.

2. Rajpurkar, Pranav & Irvin, Jeremy & Zhu, Kaylie & Yang, Brandon & Mehta, Hershel & Duan, Tony & Ding, Daisy & Bagul, Aarti & Langlotz, Curtis & Shpanskaya, Katie & Lungren, Matthew & Ng, Andrew. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. 

