# Video Classification with CNN, RNN, and PyTorch

Video classification is the task of assigning a label to a video clip. This application is useful if you want to know what kind of
activity is happening in a video. In this post, I will share a method of classifying videos using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) implemented in PyTorch.

The outline of this post is as the following:
1. TOC
{:toc}

## Introduction
A video is a collection of sequential frames or images that are played one after another. Most videos that we deal with in our daily life have
more than 30 frames per second. Thus, compared to image classification, we have to deal with a large scale of data even for short videos. Since the images are highly correlated, it is common to skip the intermediate frames and process fewer frames per second. 


## Data Preparation
The first step is to prepare the dataset. We will need a training dataset to train our model and a test or validation dataset to evaluate the model. For this purpose, we will use [HMDB: a large human motion database](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview). 

The HMDB dataset was collected from various sources, including movies, YouTube, and Google videos. It is a large dataset (2 GB) with a total of 7,000 video clips. There are 51 action classes, each containing a minimum of 101 clips. We will assign a label to each action, for example:

```python
{'sword_exercise': 0,
    'pushup': 1,
    'turn': 2,
    'draw_sword': 3,
    'clap': 4,
    ...
```

Here is the first frame of a few sample video clips:

![sample collection](/images/vidclass/samplevid.png)

You need to first download and extract the data into a local folder named data. The folder should contain 51 subfolders corresponding to 51 class actions. Also, each subfolder should contain at least 101 video files of the .avi type for each action class. 

In the first part of data preparation, we will convert the videos into images. We will only use 16 frames from each video that are equally spaced across the entire video and store them as .jpg files. This step is to reduce the computational complexity.

I defined two helper functions to get (```get_frames```) and store the frames (```store_frames```) from a video. The helper functions are defined in ```myutils.py```, which is available [here](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py).

Also, in this [notebook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/prepare_data.ipynb), you can see how I used the helper functions to loop over the videos, extract 16 frames, and store them as jpg files.

After converting the videos into images, we will split the dataset into training and test sets using ```StratifiedShuffleSplit```. 

Next, we will define a PyTorch dataset class called ```VideoDataset```. In the class, we will load all 16 images per video, down-sample them to 112 by 112, and stack them into a PyTorch tensor of shape ```[16, 3 112, 112]```.

Then, we will define two instances of the class for the training and test datasets. Next, we will define two data loaders. Data loaders will help us to automatically grab mini-batches from the dataset during training. For instance, if we set ```batch_size=8```, data loaders will return mini-batches (tensors) of shape ```[8, 16, 3, 112, 112]``` in each iteration.

See this [notebook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/Chapter10.ipynb) for the source code of the dataset and data loader classes.

## Model Implementation
We will use a deep learning model to process multiple images of a video to extract the temporal correlation. The model is a combined CNN-RNN architecture. The goal of RNN models is to extract the temporal correlation between the images by keeping a memory of past images. The block
diagram of the model is as follows:

![rnn model](/images/vidclass/rnnmodel.png)

The images of a video are fed to a CNN model to extract high-level features.
The features are then fed to an RNN layer and the output of the RNN layer is connected to
a fully connected layer to get the classification output. We will use ResNet18 pre-trained on ImageNet, as the base CNN model.

You can find an implementation of the full model class in PyTorch called ```Resnt18Rnn``` in this [notebook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/Chapter10.ipynb).

Set the model parameters and define an instance of the model class:

```python
    params_model={
        "num_classes": num_classes,
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 100,}
    model = Resnt18Rnn(params_model)      
```



## Model Training
It is time to train the model. The base CNN model was pre-trained. So we will start from the pre-trained weights and fine-tune the model on the HMDB dataset.
The training scripts can be found in [myutils.py](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py). Set the parameters and call ```train_val``` function to train the model.

```python
params_train={
"num_epochs": 20,
"optimizer": opt,
"loss_func": loss_func,
"train_dl": train_dl,
"val_dl": test_dl,
"sanity_check": True,
"lr_scheduler": lr_scheduler,
"path2weights": "./models/weights_"+model_type+".pt",
}
model,loss_hist,metric_hist = myutils.train_val(model,params_train)
```

## Model Deployment

Now, it's time to deploy the model on a video. I put the required utility functions in the [myutils.py](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/myutils.py) file. To deploy the model, we need to instantiate an object
of the model class. You can do this by calling the ```get_model``` utility function defined in ```myutils.py```. 

```python
import myutils
model_type = "rnn"
model = myutils.get_model(model_type = model_type, num_classes = 5)
model.eval();
```

Then, we will load the trained weights into the model.

```python
import torch
path2weights = "./models/weights_"+model_type+".pt"
model.load_state_dict(torch.load(path2weights))
```

Let us load a video using ```get_frames```:

```python
frames, v_len = myutils.get_frames(path2vido, n_frames=16)
```

Here are a few frames of the video:
![sample video deploy](/images/vidclass/samplevid2.png)


Then, conver the frames into a tensor and pass it to the model to get the predictions:

```python
with torch.no_grad():
    out = model(imgs_tensor.to(device)).cpu()
    print(out.shape)
    pred = torch.argmax(out).item()
    print(pred)
```

The deployment scripts can be found in this [notebook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/DeployingModel.ipynb).
