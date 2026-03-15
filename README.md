# Ishan Agrawal and Lin Hao Final Project ECE 176

The project is organised as follows:

# 1. Model definition:

is under /models/UNet/ which has all the layers we used to build the model. We have two models here which are the Attention UNet model and the regular UNet model without attention. Additionally, we have also defined 3 custom layers which we used within the network in the /blocks/ folder, which are the ConvReLU layer for downsampling, ConvTranspose for upsampling, and another upsampling layer but with attention. The models/UNet/ folder also contains a file called params.py with the dataclass containing the parameters we used to define the UNet architecture

# 2. Preprocessing:

Under the folder preprocess we have two files: conversion.py and splitting.py. In conversion.py we have many utility functions
to visualise the data in .nii format. Additionally, there is a function which we created the preprocess the data by using a MinMax scalar and cropping the data in the center to the size (128, 128, 128) and convert the labels using one-hot encoding. In splitting.py we have a function to combine all the individial samples into a single numpy array and we also have a function to generate our train and test split from all the data, before we store them as PyTorch dataloaders.

# 3. utility

In the /utility/ folder contains 3 files. metrics.py is the files we used to define our custom cross-entropy and dice score loss function. Additionally, in the file we have created utility functions to calculate a sample's intersection over union (IoU) and dice score, which we used to evaluate the models after training. Furthermore, there is a file called plotting.py which we used to do our plotting of loss curves, dice scores and segmentation masks using matplotlib. Under utils.py we defined our train loop, inference loop and custom PyTorch Dataset class for our data. Moreover, there is also a function here where we initialised the network weights 
using He initialization. 

# 4. trained_models

In this folder contains the state dict of all the models we trained for our project.

# 5. Train and Eval Notebooks

In the files train.ipynb and eval.ipynb we trained the models and evaluated them
