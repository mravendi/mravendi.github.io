# Best Practices for Building Better ML Models

Learning to apply best practices is the key to improve the performance of machine learning (ML) models.
In this post, I will share some of the best practices that apply to most ML problems. The code snippets of this post are based on Python and PyTorch.

## Basic Setup
First, let us consider a basic setup. To train ML models, we need the following elements:
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

![datasplit](/images/tipstricks/datasplit.png)

People usually  use ``` sklearn.model_selection.ShuffleSplit ``` which randomly splits the data for training and testing.
However, a key point to keep in mind is to split your data by groups if applicable. For example, assume that you have 100 images from 10 patients (10 per patient). In this case, you need to split the data patient-wise and not image-wise. 
For this type of data, I usually use  ```sklearn.model_selection.GroupShuffleSplit ```:


### Managing Experiments
When building ML models, you are going to experiment a lot by trying different hyper-parameters, models, and data. It is a good practice to keep track of your experiments. An easy method is to write a few lines of codes to automatically create a folder per experiment and store all parameters and models in the folder. You can also try using open source tools such as [MLflow](https://mlflow.org/) to manage your experiments.



### Defining Loss function
The loss function will guide the optimizer to update the model parameters. You need to define the loss function according to the task and data. 
Most often you can use standard loss functions and sometimes you have to define custom loss functions depending on the data and task. 
You can check out Chapters 4 and 5 of my [book](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830/ref=sr_1_2_sspa?dchild=1&keywords=computer+vision+cookbook&qid=1592198268&sr=8-2-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzUEVPNEI1REE4WTBQJmVuY3J5cHRlZElkPUEwODI5NjUxMlQ2T0ZCSEkxNTg4NiZlbmNyeXB0ZWRBZElkPUEwODQ0NjgxQTdaRDhYQjdXWFFQJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==) for examples of custom loss functions.


### Defining Evaluation Metrics
To make progress in your experiments, you need to define a proper evaluation metric for the problem at hand. It will give you a way to measure progress and design better experiments. Evaluation metrics are task-specific. For example, Intersection Over Union (IOU) in segmentation problems or Area Under the Curve (AUC) in classification problems are very common. 

Here is an example of defining IOU using torchvision:
```python
import torchvision
iou=torchvision.ops.box_iou(output, target)
```



### Learning Rates
The learning rate is one of the most important hyper-parameters in ML experiments. That means you need to try to find its value by experimenting. Start with default values but do not settle on that. Experiment and find the best value.

Here is an example in PyTorch to set or read the learning rate:

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
If you are familiar with overfitting, you would hate it, if not check out this [post](https://mravendi.github.io/2018/02/28/AnotherLook.html), you are going to hear a lot about overfitting. 

In a nutshell, overfitting happens when your models are over-trained and thus cannot generalize beyond the training dataset. This is how it looks like if you plot the loss values for the training and validation datasets:

![ovefitting](/images/tipstricks/overfitting.png)

That is why it is important to monitor the progress of training and validation losses and metrics during training to be able to stop the training once needed. 


### Storing Weights
ML models are fitted on the training data recursively (each iteration called an epoch). But the models do not necessarily improve in each epoch. A good practice is that
after an epoch, store the updated weights only if there is an improvement in the validation metrics. Check out the following snippet for a way of doing this method:

```python
if val_loss < best_loss:
  best_loss = val_loss
  best_model_wts = copy.deepcopy(model.state_dict())
```

### Learning Rate Schedules
When training an ML, it is normal to see that the loss function drops quickly and then stops at a certain point or plateaus. In such situations, changing the learning rate
can help the model to scape the plateau and continue with its decline. To change the learning rate, learning rate schedules have been used either manually or automatically to take care of the learning rate. The process is that we monitor the loss value on the validation data and once it reaches a plateau, we usually decrease the learning rate by some factor. There are more varieties of learning rates that you can find on the PyTorch website or real examples in my [book](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830).

![lr-schedule](/images/tipstricks/lrsch.png)

Here is a code snippet to define a learning rate schedule:

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

Here is an example of data augmentation using torchvision:

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
Instead of building a custom model and train it from scratch, we can use state-of-the-art pre-trained models and fine-tune them for our applications.
Such models were trained on a large dataset and can boost the performance of your task once fine-tuned. Here is an example of defining a resnet18 model:

```python
from torchvision import models
model_resnet18 = models.resnet18(pretrained=True)
```

You can find an example of employing pre-trained models in Chapter 3 of my [book](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830).

### Data Normalization
Data normalization is a pre-processing step that you can perform on-the-fly together with data augmentation. Usually, we perform multiple
steps of data augmentation in series on a batch of data and then normalize the outcome using one of the common techniques such as zero-mean unit variance, scale to the range of [0,1], etc. The point of data normalization is to convert the data to a standard range to help with the training. 

Here is an example of data normalization with the zero-mean unit-variance approach:
```python
train_transformer = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomVerticalFlip(p=0.5),
  transforms.ToTensor(),
  transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])])
```



### Ensembling
Finally, here comes ensembling. Ensembling multiple models is a powerful technique to boost the performance of ML systems. 
The idea is to train multiple models on a different combination of data. At deployment time, we get the output of individual models and take the average of all models as the final output.

![ensemble](/images/tipstricks/ensemble.png)

By training different models, each model will learn a different aspect of the data that can complement each other. The famous analogy in real life to understand ensembling is 
[The Blind Men, the Elephant, and Knowledge](https://en.wikisource.org/wiki/The_poems_of_John_Godfrey_Saxe/The_Blind_Men_and_the_Elephant).


If you do not have any time constraints at deployment or your models are small, ensembling will be handy. In real-time applications with time constraints, it is hard to justify the cost versus benefit of ensembling.


















