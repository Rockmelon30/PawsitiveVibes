# Capstone Project - Dog Emotion Classifier
---
Corey Jesmer

## Problem Statement
---

Can we build a model that accurately classifies images of dog faces to determine what emotion they're showing? 

## Questions
---

1. Can we build a model that accurately classifies images of dog faces and their emotions from four classes?
2. What is our baseline?
3. How will success be measured?
4. How accurate of a model can we build and how could it be improved upon?
5. What type of model gets us the best result?

## Data Dictionary
---


| Feature             | Type   | Dataset | Description                                                                       | Note |
|---------------------|--------|---------|-----------------------------------------------------------------------------------|------|
| **Filename**        | *str*  | labels.csv     | Filename of image                       |  |    
| **Label**            | *int*  | labels.csv     | Label of associated image                                                         |   0 = Angry 1 = Happy 2 = Sad 3 = Relaxed   |



### Data Used
---
1. labels.csv
2. labels_cleaned.csv
3. Kaggle image dataset
4. best_model.keras
5. backup-best-model.keras
6. best_model.weights.h5
7. model_architecture.json

### Images Used
---
1. dog_emotions_complete
2. dog_emotions_split
3. Demo Pics
4. paw background.jpg


### Links to outside resources used:

1. Kaggle Dataset of Dog Images and Labels: [link](https://www.kaggle.com/code/crn4tww/dog-emotions-classifier)
2. Streamlit [link](https://streamlit.io/)
3. My Streamlit App: [link](https://github.com/Rockmelon30/PawsitiveVibes)
4. Video on building CNN models : [link](https://www.youtube.com/watch?v=jztwpsIzEGc&t=3543s) 



### Requirements
---
- Python, Jupyter, Streamlit
- Pandas, Numpy, Matplotlib, Seaborn
- Scikit Learn Libraries:
   - StandardScaler, Metrics
   - classification_report, accuracy_score, confusion_matrix
- Tensorflow, Keras, PIL
   - Conv2D, MaxPooling2D, Flatten, Dense, Dropout
   - Sequential, EarlyStopping, VGG16, GlobalAveragePooling2D
   - RandomFlip, RandomZoom, RandomRotation, image_dataset_from_directory
   - ImageDataGenerator, Adam, L1, L2
   - Image, ImageOps

### Executive Summary
---
 This project aims to develop a machine learning model to classify images of dog emotions. 

 
### Objectives
---
1. Data Processing: Clean and preprocess the label and image data to ensure it is suitable for model training.
2. Data Exploration: Explore data for corrupt images and remove them as necessary.
3. Model Development: Develop and train machine learning models to classify images of dogs into emotion categories.
4. Evaluation: Assess the performance of the models using various metrics to ensure their accuracy and reliability.
5. Develop a streamlit app for uploading and classifying unseen images of dogs.


### Methods
---

#### **Data Cleaning**
---
The data seemed relatively clean at first glance, our labels were accurate and there were no missing values. There were 4000 images belonging to our four different classes (Angry, Happy, Sad and Relaxed.) Upon inspection of our images however, I discovered 3 corrupt images belonging to the 'Angry' class and removed them.

#### **Baseline**
---
With four balanced classes of 1000 images each, our baseline is 25%.


#### **CNN**
---
A covoluted neural net was instantiated with a total of 6 dense hidden layers, two dropout layers and a regularization of 0.001 l2. The layers were as follows:

Input(shape=(256, 256, 3)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2),
    
Conv2D(128, (2, 2), activation='relu'),
MaxPooling2D(2),
Conv2D(128, (2, 2), activation='relu'),
MaxPooling2D(2),
    
Conv2D(256, (2, 2), activation='relu'),
MaxPooling2D(2),
Conv2D(256, (2, 2), activation='relu'),
MaxPooling2D(2),
    
Conv2D(512, (2, 2), activation='relu'),
Flatten(),

Dense(256, activation='relu'),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(64, activation='relu'),
Dropout(0.5),
Dense(4, activation='softmax')

Then a dense output layer with softmax activation and 4 neurons (multiclass classification). In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 25 epochs.


#### **CNN + Data Augmentation**
---
A covoluted neural net using data augmentation (Random Flip, Rotation and Zoom) was instantiated with a total of 7 convolutional layers, 4 dense hidden layers, and no dropout layers. The layers were as follows:


Input(shape=(256, 256, 3)),
RandomFlip('horizontal_and_vertical',
RandomRotation(0.2),
RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
    
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2),
    
Conv2D(128, (2, 2), activation='relu'),
MaxPooling2D(2),
Conv2D(128, (2, 2), activation='relu'),
MaxPooling2D(2),
    
Conv2D(256, (2, 2), activation='relu'),
MaxPooling2D(2),
Conv2D(256, (2, 2), activation='relu'),
MaxPooling2D(2),
    
Conv2D(512, (2, 2), activation='relu'),
Flatten(),

Dense(256, activation='relu'),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(4, activation='softmax')

Then a dense output layer with softmax activation and 4 neurons (multiclass classification). In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 25 epochs.

#### **VGG16 with L1 and L2 regularization**
---
A covoluted neural net was instantiated with the pretrained model VGG16, and weights = ImageNet. The layers were as follows:

VGG16_base with Input Shape (224 x 224 x 3)
GlobalAveragePooling2D()
Dense(224, Relu activation)
Dense(4, Softmax activation)

Then a dense output layer with softmax activation and 4 neurons (multiclass classification). In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 25 epochs.


#### **VGG16 with AdamW optimizer**
---
A covoluted neural net was instantiated with the pretrained model VGG16, and weights = ImageNet. The layers were as follows:

VGG16_base with Input Shape (224 x 224 x 3)
GlobalAveragePooling2D()
Dense(224, Relu activation)
Dense(4, Softmax activation)

Then a dense output layer with softmax activation and 4 neurons (multiclass classification), using AdamW as the optimizer with a learning rate of 0.0001 and weight decay of 0.01. In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 25 epochs.


#### **VGG16**
---
A covoluted neural net was instantiated with the pretrained model VGG16, and weights = ImageNet. The layers were as follows:

VGG16_base with Input Shape (256 x 256 x 3)
GlobalAveragePooling2D()
Dense(256, Relu activation)
Dense(4, Softmax activation)

Then a dense output layer with softmax activation and 4 neurons (multiclass classification). In total, this provided 32,801 trainable parameters. When fitting the neural net, a batch size of 32 was used with 25 epochs.

This model performed best, resulting in a validation score of 84%.
  

#### Findings
---
  - I was able to build a model with around 84% accuracy (baseline of 25%) with minimal bias.
  - Many of the models began to aggressively overfit after a handful of epochs.
  - Due to a recent bug with versions of tensorflow 2.16.1 and greater, I was unable to run model summary on VGG16 models.
  - Similarly, I was also unable to read in my saved VGG16 models, although they run in my streamlit app in CodeSpaces without issue.
  - My model seems to struggle most classifying Sad and Relaxed emotions, perhaps because they are visually similar.

 
#### Steamlit App
---
A streamlit app was created for purposes of demonstrating the models performance. The app is a simple one that allows users to upload an image of a dog which is then classified into one of the four categories.


### **Discussion/Conclusion**
---

1. The best model I was able to achieve was with VGG16 and an accuracy score of 86%

2. The model seems to struggle most with the difference between sad and relaxed emotions


### Next steps
---

1. For improving my model further, I believe the most important next step would be to collect more data. 1000 images per class (4000 total) may not be enough.

2. In addition to gathering additional data, I'd consider adding other emotions. (Fear, for example).

3. Try other types of pretrained models.

4. Try to discover main causes of overfitting to they can be reduced.

5. Try to find a workaround for the tensorflow bug that prevented me from viewing some model summaries and reading in my saved models.