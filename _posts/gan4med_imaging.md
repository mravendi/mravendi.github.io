# GAN for Medical Imaging: Generating Images and Annotations

In this post, we are going to show a way of using generative adversarial networks (GANs) to simultaneously generate medical 
images and corresponding annotations. We use cardiac MR images for the experiment. For model development, we use Keras with theano backend.


## Introduction

Automatic organ detection and segmentation have a huge role in medical imaging applications. 
For instance, in the cardiac analysis, the automatic segmentation of the heart chambers is used 
for cardiac volume and ejection fraction calculation. One main challenge in this field is the 
lack of data and annotations. Specifically, medical imaging annotations have to be performed by 
clinical experts, which is costly and time-consuming. In this work, we introduce a method for 
the simultaneous generation of data and annotations using GANs. Considering the scarcity of data 
and annotations in medical imaging applications, the generated data and annotations using our 
method can be used for developing data-hungry deep learning algorithms.


## Data

We used the MICCAI 2012 RV segmentation challenge dataset. 
TrainingSet, including 16 patients with images and expert annotations, was used 
to develop the algorithm. We convert the annotations to binary masks with the same size as images.
The original images/masks dimensions are 216 by 256. For tractable training, 
we downsampled the images/masks to 32 by 32. A sample image and corresponding 
annotation of the right ventricle (RV) of the heart is shown below.


## Method

We use a classic GAN network with two blocks:

- Generator: A convolutional neural network to generate images and corresponding masks.
- Discriminator: A convolutional neural network to classify real images/masks from generated images/masks.

Here mask refers to a binary mask corresponding to the annotation.

The block diagram of the network is shown below.


## Algorithm Training

To train the algorithm we follow these steps:

 - Initialize Generator and Discriminator randomly.
 - Generate some images/masks using Generator.
 - Train Discriminator using the collected real images/masks (with y=1 as labels) and generated images/masks (with y=0 as labels).
 - Freeze the weights in Discriminator and stack it to Generator (figure below).
 - Train the stacked network using the generated images with y=1 as forced labels.
 - Return to step 2.


It is noted that, initially, the generated images and masks are practically garbage. 
As the models are trained, they will become more meaningful. 
Some sample generated images and masks are depicted below.



The code is shared in this [Jupiter notebook](https://nbviewer.jupyter.org/github/mravendi/AIclub/blob/master/tutorial/notebook/GAN_CMRI_32by32.ipynb)

### References:

- [DCGAN](https://github.com/rajathkmp/DCGAN)
- [How to Train a GAN?](https://github.com/soumith/ganhacks)
- [KERAS-DCGAN](https://github.com/jacobgil/keras-dcgan)
- [Keras GAN](https://github.com/mravendi/KerasGAN)
- [Keras-GAN](https://github.com/phreeza/keras-GAN)


