# GA-Capstone-Project
Classification of Human Faces by Emotion, Gender and Age using Convolutional Neural Networks (CNN)

**Business Overview**
This project aims to detect and classify human faces by emotions, gender and age, to allow companies to better customise user experience, which would in turn generate additional sales and customer loyalty. In particular, there is increasing commercial value in predicting emotions, as it allows companies to better understand emotions induced by the products that customers interact with, and how these impact customer satisfaction and their decisions to purchase eventually. The image dataset (Specs on Face (SoF) dataset) used consists of labels for four classes of emotions – neutral (no), happy (hp), sad/angry/disgusted (sd), surprised/fearful (sr), as well as for gender and age.

The emotion detection ensemble model implemented achieved an overall classification accuracy of 78.2%, which is much higher than the baseline accuracy (25% chance random guessing). The gender detection model achieved an overall classification accuracy of 93.6%, while the age detection model achieved a Mean Absolute Error (MAE) of 3.2. 

The three models were combined for deployment to a prototype Telegram bot @FaceClassificationBot using Flask/Heroku, for real-time prediction of a person’s emotions, gender and age based on the image provided.
This project was implemented primarily on the Google Colab platform, to leverage the Cloud GPU resources for the computational load required for training CNN models.

Disclaimer: Some of the ipynb files may not work directly because they were coded in Google Colab (with slightly different syntax, library imports and file paths). Comments are included in detail within the code to improve readability. Large data files are zipped to meet GitHub’s file size limitations.

**Dataset**
The SoF dataset (https://sites.google.com/view/sof-dataset) was used. It is a collection of images of 112 persons who wear glasses under different light illumination conditions. This dataset was devoted to two main problems in image classification:
- Face occlusions
- Harsh illumination environments

The SoF dataset contains image labels for four classes of emotions – neutral (no), happy (hp), sad/angry/disgusted (sd), surprised/fearful (sr), as well as for gender and age, which are required for this project. However, one disadvantage is that the SoF dataset contains too few images with sd/sr emotions; hence, the CK+48 dataset (https://www.kaggle.com/shawon10/ckplus) was subsequently added to increase the size of the train dataset for the emotion detection model.

**Data Cleaning and Pre-processing**
- Resize to standardise all image size (150x150px)
- Convert to greyscale and normalise images to reduce effect of poor lighting (also tried Adaptive Thresholding, but did not improve results)
- Image augmentation (horizontal flipping) to increase size of train dataset

**Models**
- Face Detection (transfer learning using pre-trained MTCNN model)
- Gender and Age Detection (reference from LeNet-5 CNN architecture)
- Emotion Detection (CNN ensemble with models of slightly different architectures and hyperparameters – combining Hyperas top performing models + grid search optimal weights for each model)

For the Emotion Detection model, using a simple reference model based on the LeNet-5 architecture did not produce good results; hence, further model optimisations (e.g. class weights, dropouts, batch normalization, early stopping, more filters/layers/neurons) were implemented to improve the model performance. An Ensemble was eventually chosen as well, to allow multiple models to complement each other in detection of different emotion classes. 

**Evaluation**
The emotion detection ensemble model implemented achieved an overall classification accuracy of 78.2%, which is much higher than the baseline accuracy (25% chance random guessing). For images that were wrongly classified, most were classified to "neutral" instead. 

The gender detection model achieved an overall classification accuracy of 93.6%, while the age detection model achieved a Mean Absolute Error (MAE) of 3.2. 

**Deployment**
The models developed were combined for deployment to a prototype Telegram bot @FaceClassificationBot using Flask/Heroku, for real-time prediction of a person’s emotions, gender and age based on the image provided by the user.
