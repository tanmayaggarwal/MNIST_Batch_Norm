# Batch normalization

## Overview

Batch normalization is a technique that helps deep learning models converge faster and result in more accuracy. Instead of just normalizing the inputs to the network, we normalize the inputs to every layer within the network. Specifically, batch normalization normalizes the output of a previous layer by subtracting the batch mean and dividing by the batch standard deviation.

## How to implement batch normalization

Batch normalization is implemented within the __init__ function when defining the model architecture. When using batch normalization, we do not include bias in the layers. Batch normalization is applied before the activation function of each hidden layer.

For linear networks, nn.BatchNorm1d is used.

A small improvement is noted when comparing batch normalized model's accuracy in evaluation mode (relative to training mode). This is because in training mode, the batch normalization layers use batch statistics to calculate the batch norm. On the other hand, during evaluation mode, the layers use the estimated population mean and variance from the entire training set, resulting in increased performance.

Models that use batch normalized layers show a marked improvement in overall accuracy when compared with the no-normalization models.

For Convolutional Neural Networks, nn.BatchNorm2d is used.

Convolution layers consist of multiple feature maps, and the weights for each feature map are shared across all the inputs that feed into the layer. Thus, batch normalizing convolutional layers requires batch/population mean and variance per feature map rather than per node in the layer.

For Recurrent Neural Networks, implementation of batch normalization involves calculating the means and variances per time step instead of per layer.

## Overview this repository

The batch_normalization_example.py file contains an example implementation of batch normalization on MNIST dataset using a linear neural network. Results compare the performance of batch normalized network vs. non-normalized network.
