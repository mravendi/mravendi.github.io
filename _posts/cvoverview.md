
## An overview of Major Computer Vision Applications

Computer vision (CV) has been disrupted by deep learning and convolutional neural networks(CNN) in recent years. You can now implement many CV algorithms pretty quickly using deep learning libraries such as PyTorch, Tensorflow, and Keras. In this post, I will provide an overview of major CV tasks and applications. You can find PyTorch implementations of these tasks with step-by-step explanations in my book [PyTorch Computer Vision Cookbook](https://www.amazon.com/PyTorch-Computer-Vision-Cookbook-computer/dp/1838644830/ref=sr_1_2_sspa?dchild=1&keywords=computer+vision+cookbook&qid=1592198268&sr=8-2-spons&psc=1&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUEzUEVPNEI1REE4WTBQJmVuY3J5cHRlZElkPUEwODI5NjUxMlQ2T0ZCSEkxNTg4NiZlbmNyeXB0ZWRBZElkPUEwODQ0NjgxQTdaRDhYQjdXWFFQJndpZGdldE5hbWU9c3BfYXRmJmFjdGlvbj1jbGlja1JlZGlyZWN0JmRvTm90TG9nQ2xpY2s9dHJ1ZQ==). 

The implementation scripts are available in this [link](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook).

The outline of this post is as the following:
1. TOC
{:toc}

## Image classification
Image classification (also called image recognition) is probably the most widely used task in computer
vision. In this task, we assume that images contain a main object and we want to automatically classify the object into pre-defined categories. In the context of image recognition, you will find the binary classification
and multi-class classification. 

### Binary Classification
In binary image classification, we want to classify images into two categories. For
instance, we may want to know if a medical image is normal or malignant. Thus, you can assign label=0 to a normal image and 
label=1 to a malignant image. That is why it is called the binary classification. In the following, you can see an example of binary image classification for image patches of Histopathologic Cancer.

![Binary image classification](/images/cv_overview/binary_image_classification.png)
*Sample image patches of Histopathologic Cancer and their binary labels*

Typically, there are thousands of these patches per patient and clinicians have to go through them one by one. Just imagine
how an automatic tool to quickly label thousands of such images can be beneficial to clinicians.

You can learn to implement a binary image classification using PyTorch in Chapter 2 of my book.
 

### Multi-class Classification
On the other hand, the goal of multi-class image classification is to automatically assign a label to an image from a fixed (more than two) set of categories. Again, here the assumption is that the image contains a dominant object. 
For instance, the following figure shows a few samples from a dataset with 10 categories.

![MultiClass](/images/cv_overview/multiclass.png)

We may assign label 5 to dogs, label 2 to cars, label 0 to airplanes, and label 9 to trucks. As you may note, there may be more than
one object in the images, however, the labels correspond to the dominant objects.

This task has also many applications in the industry, from autonomous vehicles to medical imaging, to automatically identify objects in images. In Chapter 3 of my book, you can learn to develop a multi-class classification model in PyTorch.

## Object Detection
Object detection is the process of finding locations of specific objects in images. Similar to image classification, depending on the number of objects in images, we may deal with single-object or multi-object
detection problems. 

### Single-object Detection
In single-object detection, we are interested to find the location of an object in a given image. In other words, we know the class of the object and only want to locate it in the image. The location of the object can be defined by a bounding box using four numbers, specifying the coordinates of the top left and bottom right corners.

As an example, the following image depicts the location of the fovea (a small pit) in an eye image using a green bounding box:

![MultiClass](/images/cv_overview/singleobject.png)

You can learn to implement a single-object detection model in PyTorch from Chapter 4 of my book.


### Multi-object Detection
On the other hand, multi-object detection is the process of locating and classifying existing objects in an image. In other words, it is a simultaneous classification and localization task.
Identified objects are shown with bounding boxes in the image, as shown in the following figure. 

![MultiClass](/images/cv_overview/multiObjectDetection.png)

As you can see, each object is identified and labeled with a category label and located by a bounding box.

Two methods for general object detection include region proposal-based and regression/classification-based. In Chapter 5 of my book, you can learn to develop a regression/classification-based object detection algorithm using PyTorch.


## Image Segmentation
Object segmentation is the process of finding the boundaries of target objects in images. There are many applications for segmenting objects in images. As an example, by outlining anatomical objects in medical images, clinical experts can learn useful information
about patients.

Depending on the number of objects in images, we can deal with single-object or multi-object segmentation tasks. 

### Single-object Segmentation 
In single-object segmentation, we are interested in automatically outlining the boundary of one target object in an image.
The boundary of the object is usually defined by a binary mask. From the binary mask, we can overlay a contour on the image to outline the object boundary. As an example, the following figure depicts an ultrasound image of a fetus, a binary mask corresponding
to the fetal head, and the segmentation of the fetal head overlaid on the ultrasound image:

![MultiClass](/images/cv_overview/segmentation.png)

The goal of automatic single-object segmentation will be to predict a binary mask
given an image. In Chapter 6 of my book, you will learn to implement a deep learning algorithm for single-object segmentation using PyTorch.


### Multi-object Segmentation
On the other hand, in multi-object segmentation, we are interested in automatically outlining
the boundaries of multiple target objects in an image. The boundaries of objects in an image are usually defined by a segmentation mask that's the same size as the image. In the segmentation mask, all the pixels that belong to a target
object are labeled the same based on pre-defined labeling. For instance, in the following
screenshot, you can see a sample image with two types of targets: babies and chairs.

![MultiClass](/images/cv_overview/msegmentation.png)

The corresponding segmentation mask is shown in the middle of the figure. As we can see,
the pixels belonging to the babies and chairs are labeled differently and colored in yellow
and green, respectively.

The goal of multiple-object segmentation will be to predict a segmentation mask given
an image such that each pixel in the image is labeled based on its object class. In 
Chapter 7 of my book, you will learn to develop a multi-object segmentation algorithm using PyTorch. 


## Style Transfer
You want to do something fun with images. Try neural style transfer. In neural style transfer, we take a regular image called the content image, and an artistic image called the style image. Then, we generate an image to have the content of the content image and the artistic style of the style image.
By using the masterpieces of great artists as the style image, you can 
generate interesting images using this technique.

As an example, check out the following figure:
![MultiClass](/images/cv_overview/styletransfer.png)

The image on the left is converted to the image on the right using a style image (middle).
In Chapter 8 of my book, you can learn to implement the neural style transfer algorithm using PyTorch. 


## GANs
Do you want more fun with images? Try GANs. A GAN is a framework that's used to generate new data by learning the distribution of data. 
The following figure shows a block diagram of a GAN for image generation.

![MultiClass](/images/cv_overview/gan.png)

The generator generates fake data when given noise as
input, and the discriminator classifies real images from fake images. During training, the generator and the discriminator compete with each other in a game. The generator tries to generate better-looking images to fool the
discriminator, and the discriminator tries to get better at identifying real images from fake
images.

In Chapter 9 of my book, you will learn to develop a GAN to generate new images using PyTorch.

## Video Classification
Images are still and static. There is no motion in static images. The real joy comes from
the motion. And that is how videos come into play. A video is, in fact, a collection of sequential frames or images that are played one after another. Check out a short clip of Matrix (the movie) in the next figure.

![MultiClass](/images/cv_overview/video.png)

Similar to image classification, you can think of video classification. Due to a large number of frames in videos, the task is daunting but doable with the help of deep learning and PyTorch. Video classification is about understanding the activity happening in videos. In Chapter 10 of my book, you can learn to build a video classification algorithm using PyTorch.


