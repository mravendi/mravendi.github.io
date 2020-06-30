# Some Best Practices for Building Better ML Models

Learning to apply best practices is the key to improve the performance of machine learning (ML) models.
In this post, I will share some of the best practices that apply to most of the ML problems. The code snippets of this post are based on Python and PyTorch.

## Basic Setup
To train ML models, we need the following elements:
- data 
- a model 
- a loss function 
- an optimizer

We use the data to train a model by defining a loss function and optimizing the model parameters to minimize an error.


### Fixing Random Seed Points
One of the simple yet important steps when building ML models is to fix the random seed point in your ML experiments. In a typical ML experiment, we perform multiple steps that
rely on random distributions such as data shuffling, data splitting, weight initialization, etc. Due to randomness, we may see different results with the same set of hyper-parameters if we do not fix the random seed. Therefore, for the sake of reproducibility and making better sounding conclusions when doing ML experiments, you can add the following snippet right at the beginning of your scripts.

```python
import numpy as np
import random
import torch

seed = 2020 # an optional integer number
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

```


### Splitting Data
If you have not previously, split your data into three groups: training, validation, and
test datasets. We use the training dataset to train the model. The validation dataset is used to track
the model's performance during training. We use the test dataset for the final evaluation of the model. 

A key point to keep in mind is to split your data by group or source. For example, assume that you have 100 images from 10 patients (10 per patient). In this case, you need to split the data patient-wise and not image-wise. For this, I usually use the ```group_kfold ``` method from ```sklearn```. Here is an example:

```python
 import numpy as np
from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=n_splits)
 group_kfold.get_n_splits(X, y, groups)
```


### Managing Experiments
When building ML models, you are going to experiment a lot by trying different hyper-parameters, models, and data. It is a good practice to keep track of your experiments. An easy method is to write a few lines of codes to automatically create a folder per experiment and store all parameters and models in the folder. You can also try using open source tools such as (MLflow)[https://mlflow.org/] to manage your experiments.



### Defining the Loss function
The loss function will guide the optimizer to update the model parameters. Define the loss function according to the task and data. Most often you can use standard loss functions and sometimes you have to define custom loss functions depending on the data and task. 


### Defining Evaluatuion Metrics
Define a proper evaluation metric for the problem at hand. For example, Intersection Over Union (IOU) in segmentation problems or Area Under the Curve (AUC) in classification problems are common. Here is an example of defining IOU using PyTorch:

```python
import torchvision
iou=torchvision.ops.box_iou(output, target)
```

### Learning Rates
The learning rate is one of the most important hyper-parameters in ML experiments. When we say hyper-parameter, that means you need to try to find its value by experimenting.
For instance, if you are training a CNN model from scratch with the weights randomly initialized, you need a bigger learning rate (in the order of 1e-4) compared to when you are fine-tuning a model on a pre-trained model (in the order of 10e-6). 

Here is an example in PyTorch to set or change the learning rate:

```python
from torch import optim
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
```

To read the current value of the learning rate, you can do:

```python

def get_lr(opt):
  for param_group in opt.param_groups:
    return param_group['lr']

current_lr=get_lr(opt)
print('current lr={}'.format(current_lr))
current lr=0.0003
```


### Monitoring Metrics and Early stopping
If you are familiar with overfitting, you hate it if not you are going to hear a lot about it. Overfitting happens when your models are over-trained and thus cannot generalize beyond the training dataset. This is how it looks like if you plot the loss values for the training and validation datasets:

![ovefitting](https://github.com/mravendi/mravendi.github.io/blob/master/images/tipstricks/overfitting.png)

That is why it is important to monitor the progress of training and validation losses and metrics during training to be able to stop the training once needed. 


### Storing Good Weights
ML models are fitted on the training data recursively (each iteration called an epoch). But the models do not necessarily improve in each epoch. A good practice is that
after an epoch, store the updated weights only if there is an improvement in the validation metrics. Check out the following snippet for a way of doing this method:

```python
if val_loss < best_loss:
  best_loss = val_loss
  best_model_wts = copy.deepcopy(model.state_dict())

torch.save(model.state_dict(), path2weights)
```

### Learning Rate Schedules
When training an ML, it is normal to see that the loss function drops quickly and then stops at a certain point or plateaus. In such situations, changing the learning rate
can help the model to scape the plateau and continue with its decline. To change the learning rate, learning rate schedules have been used either manually or automatically to take care of the learning rate. The process is that we monitor the loss value on the validation data and once it reaches a plateau, we usually decrease the learning rate by a factor of 2. There are more varieties of learning rates that you can find on PyTorch website or real examples in my [book](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830/ref=sr_1_1_sspa?crid=357W25TVH92GN&dchild=1&keywords=pytorch+computer+vision+cookbook&qid=1592800424&sprefix=pytocrch+comp%2Caps%2C201&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExTVlaS1VQVTQ5TUpMJmVuY3J5cHRlZElkPUEwMDc5NzE1U0xQVktER1FOVkMwJmVuY3J5cHRlZEFkSWQ9QTA4NDQ2ODFBN1pEOFhCN1dYUVAmd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl).

In PyTorch, it is very easy to define a learning rate schedule. Here is a snippet:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)
```
The above learning rate schedule would wait for ```patience=20``` epochs before halving the learning rate.


### Data Augmentation
Data augmentation is another critical step in training ML  algorithms, especially for small datasets to reduce/avoid overfitting. It can help 
to reduce the overfitting by artificially enlarging your training dataset. 
Even if you think that your training dataset is big enough, data augmentation can help with the robustness of your models. 
The process of data augmentation consists of randomly transforming the original data to get new data. 
You can use ready-to-use Python packages to perform data augmentation on-the-fly. 

For instance, in image classification/detection tasks, you can randomly flip images horizontally as seen in the following figure:

![dataaugmentation](/images/tipstricks/dataaug.png)

Here is an example of how to do data augmentation using torchvision module of PyTorch:

```python
train_transformer = transforms.Compose([
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomVerticalFlip(p=0.5),
                              transforms.RandomRotation(45),
                              transforms.RandomResizedCrop(96,scale=(0.8,1.0),ratio=(1.0,1.0)),
                              transforms.ToTensor()])
```

### Pre-Trained Models
One of the successful techniques in developing ML models is the use of pre-trained models and transfer learning. 
Instead of building a custom model and train it from scratch, we can use state-of-the-art pre-trained models for our applications.
Such models were trained on a large dataset and can boost the performance of your task. Here is an example of defining a resnet18 model:

```python
from torchvision import models
model_resnet18 = models.resnet18(pretrained=False)
```

### Data Normalization
Data normalization is a pre-processing step that you can perform on-the-fly together with data augmentation. Usually, we perform multiple
steps of data augmentation in series on a batch of data and then normalize the outcome using one of the common techniques such as zero-mean unit variance, scale to the range of [0,1], etc. The point of data normalization is to change the range of your data to a standard range to help with the training. 

### Model Deployment
After training your ML models, you certainly want to deploy them for an application. Do not forget to perform the same pre-processing steps used for training during deployment. For example, if you scaled your training data to the range of [0,1], you need to do the same during deployment.  





