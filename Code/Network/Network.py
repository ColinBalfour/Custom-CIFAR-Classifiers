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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(F.log_softmax(outputs,dim=1), dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = nn.CrossEntropyLoss()(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))



class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      
      self.conv = nn.Sequential(
            nn.Conv2d(InputSize, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

      self.model = nn.Sequential(
          nn.Linear(3136, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, OutputSize),
          # nn.Softmax(1),
      )
      

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################
    #   print(xb.shape)
      
      x = self.conv(xb)
      x = torch.flatten(x, 1)
      return self.model(x)


class CIFAR10ModelImproved(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      
      self.conv = nn.Sequential(
            nn.Conv2d(InputSize, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # add two more layers with max pooling and different kernel size, stride, padding
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

      self.model = nn.Sequential(
          nn.Linear(2304, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, OutputSize),
          # nn.Softmax(1),
      )
      

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################
    #   print(xb.shape)
      
      x = self.conv(xb)
    #   print(x.shape)
      x = torch.flatten(x, 1)
      return self.model(x)


class ResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, first_stride=1):
      super().__init__()
      self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels)
      )

      if in_channels != out_channels:
        self.skip = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride),
          nn.BatchNorm2d(out_channels)
        )
      else:
        self.skip = nn.Identity()

  def forward(self, xb):
      # print(xb.shape)
      out = self.conv1(xb)
      out += self.skip(xb)
      out = nn.ReLU()(out)
      # print(xb.shape, out.shape)
      return out

class CIFAR10ModelResNet(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      
      self.conv1 = nn.Sequential(
            nn.Conv2d(InputSize, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
      )
      
      self.resnet1 = nn.Sequential(
        ResNetBlock(64, 64, first_stride=1),
        ResNetBlock(64, 64),
      )
      
      self.resnet2 = nn.Sequential(
        ResNetBlock(64, 128, first_stride=2),
        ResNetBlock(128, 128),
      )
      
      self.resnet3 = nn.Sequential(
        ResNetBlock(128, 256, first_stride=2),
        ResNetBlock(256, 256),
      )
      
      self.resnet4 = nn.Sequential(
        ResNetBlock(256, 512, first_stride=2),
        ResNetBlock(512, 512),
      )

      self.model = nn.Sequential(
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, OutputSize),
          # nn.Softmax(1),
      )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################
    #   print(xb.shape)
      
      x = self.conv1(xb)
      x = self.resnet1(x)
      x = self.resnet2(x)
      x = self.resnet3(x)
      x = self.resnet4(x)
      x = torch.flatten(x, 1)
      return self.model(x)





class ResNeXtBlock(nn.Module):
  def __init__(self, in_channels, out_channels, first_stride=1, C=16):
      super().__init__()
      self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1, groups=C),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=C),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
      )

      if in_channels != out_channels:
        self.skip = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride),
          nn.BatchNorm2d(out_channels)
        )
      else:
        self.skip = nn.Identity()

  def forward(self, xb):
      out = self.conv1(xb)
      out = out + self.skip(xb)
      out = nn.ReLU()(out)
      # print(xb.shape, out.shape)
      return out

class CIFAR10ModelResNeXt(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      #############################
      # Fill your network initialization of choice here!
      #############################
      super().__init__()
      
      self.conv1 = nn.Sequential(
            nn.Conv2d(InputSize, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
      )
      
      self.resnext1 = nn.Sequential(
        ResNeXtBlock(64, 64, first_stride=1),
        ResNeXtBlock(64, 64),
      )
      
      self.resnext2 = nn.Sequential(
        ResNeXtBlock(64, 128, first_stride=2),
        ResNeXtBlock(128, 128),
      )
      
      self.resnext3 = nn.Sequential(
        ResNeXtBlock(128, 256, first_stride=2),
        ResNeXtBlock(256, 256),
      )
      
      self.resnext4 = nn.Sequential(
        ResNeXtBlock(256, 512, first_stride=2),
        ResNeXtBlock(512, 512),
      )

      self.pool = nn.AdaptiveAvgPool2d((1, 1))
  
      self.model = nn.Sequential(
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, OutputSize),
          # nn.Softmax(1),
      )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################
    #   print(xb.shape)
      
      x = self.conv1(xb)
      x = self.resnext1(x)
      x = self.resnext2(x)
      x = self.resnext3(x)
      x = self.resnext4(x)
      x = self.pool(x)
      x = torch.flatten(x, 1)
      return self.model(x)


class DenseNetLayer(nn.Module):
  def __init__(self, in_channels, growth_rate):
      super().__init__()
      self.dense_layer = nn.Sequential(
        nn.BatchNorm2d(in_channels),  # The input channels here should reflect the growing size
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
      )
  
  def forward(self, xb):
      out = self.dense_layer(xb)
      return out

class DenseNetBlock(nn.Module):
  def __init__(self, in_channels, growth_rate, num_layers):
      super().__init__()
      self.layers = nn.ModuleList([DenseNetLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

  def forward(self, xb):
      for layer in self.layers:
          out = layer(xb)
          xb = torch.cat([xb, out], dim=1)  # Concatenate new output to the input tensor
      return xb

class DenseNetTransitionBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
      super().__init__()
      self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.AvgPool2d(2, 2)
      )

  def forward(self, xb):
      out = self.conv1(xb)
      return out

class CIFAR10ModelDenseNet(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super().__init__()
      
      self.conv1 = nn.Conv2d(InputSize, 64, kernel_size=3, stride=1, padding=1)
      self.dense1 = DenseNetBlock(64, 32, 4)  # Input 64 channels, growth rate 32, 4 layers
      self.trans1 = DenseNetTransitionBlock(64 + 32 * 4, 128)  # Adjusted for the correct number of channels
      self.dense2 = DenseNetBlock(128, 32, 4)  # Next block
      self.trans2 = DenseNetTransitionBlock(128 + 32 * 4, 256)  # Adjusted for the correct number of channels

      self.model = nn.Sequential(
          nn.Linear(16384, 1024),  # Adjust the input size accordingly
          nn.ReLU(),
          nn.Linear(1024, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.Linear(128, OutputSize),
      )
      
  def forward(self, xb):
      x = self.conv1(xb)
      x = self.dense1(x)
      x = self.trans1(x)
      x = self.dense2(x)
      x = self.trans2(x)
      x = torch.flatten(x, 1)
      return self.model(x)
