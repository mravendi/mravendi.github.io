# ML Tips and Tricks

The machine learning and data science fields are based on some fundamentals and many tips and tricks. 
In this post, I will share some of the important techniques that can help improve the performance of your ML models.


## Basic Setup
Typically, to develop an ML/DL algorithm using SGD, the following ingredients are required:

- Training data (inputs and targets)
- a model 
- an objective function
- an optimizer

You can see the interaction between these elements in the following diagram:
![trainingloop](https://github.com/mravendi/mravendi.github.io/blob/master/images/tipstricks/trainingloop.png)



### Fixing Random Seed Points
One of the simple yet important steps when building ML models is to fix the random seed point in your ML experiments. In a typical ML experiment, we perform multiple steps that
rely on random distributions such as: data shuffling, data spliting, weight initialization, etc. Due to these randomness, we may see different results with the same set of hyper-parameters if we do not fix the random seed. Therefore, for the sake of reproducibility and making better sounding colclusions when doing ML experiments, try to add simple lines of codes right in the beggining of your scripts.

```python
import numpy as np
import random
import torch

seed = 2020 # an optional integer number
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

```

### Learning Rates
Learning rate is one of the most improtant hyper-parameters in ML experiments. When we say hyper-parameter, that means you need to try to find its value by experimenting.
Nevertheless, there are rules of thumbs and previous reported values in the literature that you can start with. 
For instance, if you are training a CNN model from scratch with the weights randomly initialized, you need a bigger learning rate (in the order of 1e-4) compared to when you are fine-tunning a model on a pre-trained model (in the order of 10e-6). For most CNN models, ``` lr= 3e-4``` works best for models trained from scratch. 

### Early stopping
Perhaps you are familiar with overfitting. It happens when your models are over-trained and thus cannot generalize beyond the training dataset. You can easily see this 
behaviour if you plot the loss values of training and validation metrics. An example is shown in the following figure.

![ovefitting](https://github.com/mravendi/mravendi.github.io/blob/master/images/tipstricks/overfitting.png)


Early stopping is the most important technique to avoid overfitting. In early stoping technique, you monitor the validation loss values and if it plateus or starts to increases
you stop the training process to avoid overfitting.

### Storing Weights During Training
Another technique to avoid overfitting, is to avoid storing/deploying an overfitted model. Thus, a good practive is that after an epoch, store the updated weights only if 
there is an improvement in the validation metrics. 

### Learning Rate Schedules
When training an ML, it is normal to see that the loss function drops quickly and then stops at a certain point or plateus. In such situations, changing the learning rate
can help the model to scape the plateu and continoue with it decline. In order to change the learning rate, learning rate schedules have been used either manually or automatically to take care of the learning rate. The process is that you monitor the loss value on the validation data and once it reaches a plateu, we usually descrease 
the learning rate by a factor of 2. There are more varieties of learning rates that you can find on PyTorch website or real examples in my [book](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830/ref=sr_1_1_sspa?crid=357W25TVH92GN&dchild=1&keywords=pytorch+computer+vision+cookbook&qid=1592800424&sprefix=pytocrch+comp%2Caps%2C201&sr=8-1-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUExTVlaS1VQVTQ5TUpMJmVuY3J5cHRlZElkPUEwMDc5NzE1U0xQVktER1FOVkMwJmVuY3J5cHRlZEFkSWQ9QTA4NDQ2ODFBN1pEOFhCN1dYUVAmd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl).









