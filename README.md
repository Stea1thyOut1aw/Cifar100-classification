# CIFAR-100 Classification with CNN

This repository contains the code for classifying images from the CIFAR-100 dataset using a Convolutional Neural Network (CNN) model. The project aims to train a CNN model to achieve high accuracy on the CIFAR-100 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The goal of this project is to classify images from the CIFAR-100 dataset using a CNN model. We implement a CNN architecture, train it on the dataset, and calculate the accuracy of the newly trained model.

## Dataset

The CIFAR-100 dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). It consists of 100 classes, each containing 600 images. There are 500 training images and 100 testing images per class.

## Installation

To run the code in this repository, you'll need to have Python installed along with several Python packages. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your-username/cifar100-classification.git
cd cifar100-classification
```

2. Download the CIFAR-100 dataset and place the train and test data in the data/ directory.
3. Train the CNN model:

```bash
python train.py
```

4. Evaluate the CNN model and calculate accuracy

```bash
python evaluate.py
```
## Model
This project uses a Convolutional Neural Network (CNN) to classify images from the CIFAR-100 dataset. The architecture includes multiple convolutional layers, pooling layers, and fully connected layers.

Key features of the CNN model:

Convolutional layers for feature extraction.
Pooling layers for down-sampling.
Fully connected layers for classification.
Softmax layer for outputting probabilities for each class.

## Results
The performance of the CNN model is evaluated on the CIFAR-100 test set. The accuracy of the newly trained model is calculated and displayed. Detailed accuracy metrics and loss curves are generated for analysis.
