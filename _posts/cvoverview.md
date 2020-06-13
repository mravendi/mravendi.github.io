
## Common Computer Vision Tasks

Computer vision has been disrupted by deep learning and convolutional neural networks. You can now implement many CV algorithms
using power deep learning libraries such as PyTorch. In this article, we will provide an overview of some of major CV tasks.


## Image classification
Image classification (also called image recognition) is probably the most widely used task in computer
vision. In this task, we assume that images contain a main object and we want to automatically 
classify the object into pre-defined categories. In the context of image recognition, you will encounter with: binary classification
and multi-class classification. 

### Binary Classification
In binary image classification, we want to classify images into two categories. For
instance, we may want to know if a medical image is normal or malignant. Thus, you can assign label=0 to a normal image and 
label=1 to a malignant image. That is why it is called binary classification. In the following, you can see an example of 
binary image classification for image patches of Histopathologic Cancer.

![Binary image classification](/images/cv_overview/binary_image_classification.png)
*Sample image patches of Histopathologic Cancer and their binary labels*

### Multi-class Classification
On the other hand, the goal of multi-class image classification is to assign a label to an image from a fixed (more than two) set of
categories. Again, here the assumption is that the image contains a dominant object. 
For instance, the following screenshot shows a few samples from a dataset with 10 categories. We may
assign label 5 to dogs, label 2 to cars, label 0 to airplanes, and label 9 to trucks. As you may note, there may be more than
one object in the images, however, we labeled the the images with their dominant objects.
### a fig here

## Object Detection
Object detection is the process of finding locations of specific objects in images. Depending
on the number of objects in images, we may deal with single-object or multi-object
detection problems. In single-object detection, we are attempting to
locate only one object in a given image. The location of the object can be defined by a
bounding box.

As an example, the following screenshot depicts the location of the fovea (a small pit) in an
eye image using a green bounding box:
### a fig here

On the other hand, multi-object detection is the process of locating and classifying existing objects in an image.
Identified objects are shown with bounding boxes in the image. There are two main methods for
general object detection: region proposal-based and regression/classification-based. 
### a fig here


## Image Segmentation
Object segmentation is the process of finding the boundaries of target objects in
images. There are many applications for segmenting objects in images. As an example, by
outlining anatomical objects in medical images, clinical experts can learn useful information
about patients' conditions.

Depending on the number of objects in images, we can deal with single-object or multiobject
segmentation tasks. In single-object segmentation, we are
interested in automatically outlining the boundary of one target object in an image.

The object boundary is usually defined by a binary mask. From the binary mask, we can
overlay a contour on the image to outline the object boundary. As an example, the
following screenshot depicts an ultrasound image of a fetus, a binary mask corresponding
to the fetal head, and the segmentation of the fetal head overlaid on the ultrasound image:
### a fig here

On the other hand, in multi-object segmentation, we are interested in automatically outlining
the boundaries of multiple target objects in an image.

The boundaries of objects in an image are usually defined by a segmentation mask that's
the same size as the image. In the segmentation mask, all the pixels that belong to a target
object are labeled the same based on a pre-defined labeling. For instance, in the following
screenshot, you can see a sample image with two types of target: babies and chairs. The
corresponding segmentation mask is shown in the middle of the screenshot. As we can see,
the pixels belonging to the babies and chairs are labeled differently and colored in yellow
and green, respectively.
### a fig here

## Style Transfer
In neural style transfer, we take a content image and a style image. Then, we generate an
image to have the content of the content image and the artistic style of the style image.
By using the masterpieces of great artists as the style image, you can 
generate very interesting images.
### a fig here

## GANs
A GAN is a framework that's used to generate new data by learning the distribution of data. In the context of image generation, the generator generates fake data, when given noise as
input, and the discriminator classifies real images from fake images. During training, the
generator and the discriminator compete with each other in a game and as a result, get
better at their jobs. The generator tries to generate better-looking images to fool the
discriminator and the discriminator tries to get better at identifying real images from fake
images.
### a fig here


