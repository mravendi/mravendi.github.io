# Video Processing with Deep Learning and PyTorch

Video classification is a task assigning a label to a video clip. In this post, I will share a method of classifying videos using deep CNN.


## Introduction
A video is, collection of sequential frames or images that are played one after another. Most of the videos that we deal with in our daily life have
more than 30 frames per second. Thus, compared to image classification, we have to deal with a large scale of data even for short videos.


## Dataset
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


### Preparing the data
In the first part of data preparation, we will only use 16 frames from each video that are equally spaced across
the entire video and store them as .jpg files. To this end, I defined two helper functions to get and store the frames from a video. The scripts 
is available in this [link](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py).

















