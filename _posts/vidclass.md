# Video Processing with Deep Learning and PyTorch

Video classification is a task assigning a label to a video clip. In this post, I will share a method of classifying videos using deep CNN.


## Introduction
A video is, collection of sequential frames or images that are played one after another. Most of the videos that we deal with in our daily life have
more than 30 frames per second. Thus, compared to image classification, we have to deal with a large scale of data even for short videos.


## Data Preparation
The first step is to create the dataset. We will need a training dataset to train our
model and a test or validation dataset to evaluate the model. For this purpose, we will use
HMDB: a large human motion database, available [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview). 

The HMDB dataset was collected from various sources, including movies, the Prelinger
archive, YouTube, and Google videos. It is a pretty large dataset (2 GB) with a total of 7,000
video clips. There are 51 action classes, each containing a minimum of 101 clips.

Let us check out a few sample actionsin the following image:

![sample collection](/images/vidclass/samplevid.png)

First download and extract the data into a local folder named data. The folder should
contain 51 subfolders corresponding to 51 class actions. Also, each subfolder should contain
at least 101 video files of the .avi type for each action class. 


To create a dataset for video classification, we will convert the videos into images. Each
video has hundreds of frames or images. To reduce computational complexity, we will select 16 frames per video that are equally
spaced across the video. Then, we will define a PyTorch dataset class and a dataloader.


In the first part of data preparation, we will only use 16 frames from each video that are equally spaced across
the entire video and store them as .jpg files. To this end, I defined two helper functions to get (```get_frames```) and store the frames (```store_frames```) from a video. The helper functions are defined in ```myutils.py```, which is available [here](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py).

Also, in this [notebook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/prepare_data.ipynb), you can see how I used the helper functions to loop over the videos, extract 16 frames and store them as jpg files.

After converting the videos into images, we will split the dataset into training and test sets using ```StratifiedShuffleSplit```. Next, we will define a PyTorch dataset class called ```VideoDataset```. In the class, we will load all 16 images per video, down-sample them to 112 by 112 and stack them into a PyTorch tensor of shape ```[3, 16, height=112, width=112]```.

Then, we will instantiate two objects of the class for the training and test datasets. Next, we will define two data loaders. Data loaders will help us to automatically 
grab mini-batches from the dataset during training. For instance, if we set batch_size=8, data loaders will return mini-batchs (tensors) of shape ```[8, 3, 16, 112, 112]``` in each iteration.



## Building the Model
We will use a model to process multiple images of a video in order
to extract temporal correlation. The model is based on RNN architecture. The goal of RNN models is to extract the
temporal correlation between the images by keeping a memory of past images. The block
diagram of the model is as follows:

![rnn model](/images/vidclass/rnnmodel.png)

As we can see, the images of a video are fed to a base model to extract high-level features.
The features are then fed to an RNN layer and the output of the RNN layer is connected to
a fully connected layer to get the classification output. The input to this model should be in
the shape of [batch_size, timesteps, 3, height, width], where timesteps=16 is
the number of frames per video. We will use one of the most popular models that has been
pre-trained on the ImageNet dataset, called ResNet18, as the base model.



























