#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import torchvision
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import CIFAR10Model, CIFAR10ModelImproved, CIFAR10ModelResNet, CIFAR10ModelResNeXt, CIFAR10ModelDenseNet
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchsummary import summary


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    
    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue, y_pred=LabelsPred)

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    # Print the accuracy
    print('Accuracy: ' + str(Accuracy(LabelsPred, LabelsTrue)) + '%')
    
    # Plot the confusion matrix using matplotlib.
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add labels, ticks and titles to the plot
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Add text annotations on each cell
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = CIFAR10ModelDenseNet(InputSize=3,OutputSize=10) 
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    
    summary(model.to('cuda'), (3, 32, 32))
    model.to('cpu')
    model.eval()
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
            
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    
    OutSaveT = open(LabelsPathPred, 'w')

    for count in tqdm(range(len(TestSet))): 
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)

        PredT = torch.argmax(model(torch.tensor(Img))).item()

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    # Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/aa/144model.ckpt', help='Path to load latest model from, Default:ModelPath')
    # Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    # ModelPath = Args.ModelPath
    TrainLabel = './TxtFiles/LabelsTrain.txt'
    TestLabel = './TxtFiles/LabelsTest.txt'
    TrainSet = torchvision.datasets.CIFAR10(root='data/', train=True, transform=ToTensor())
    TestSet = torchvision.datasets.CIFAR10(root='data/', train=False,transform=ToTensor())

    CheckPointPath = 'DenseNetCheckpoints/12model.ckpt'
    BasePath ='CIFAR10'


    # Setup all needed parameters including file reading
    ImageSize = SetupAll()
    
    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels
    
    ## TRAIN SET ##

    TestOperation(ImageSize, CheckPointPath, TrainSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(TrainLabel, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)
    
    ## TEST SET ##
    
    TestOperation(ImageSize, CheckPointPath, TestSet, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(TestLabel, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
