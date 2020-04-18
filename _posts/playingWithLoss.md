# Playing with Loss Functions in Deep Learning

In this post, we are going to be developing custom loss functions in deep learning applications
such as semantic segmentation. We use Python 2.7 and Keras 2.x for implementation. Here is the outline of this post:

1. TOC
{:toc}

## Standard Loss Function
Loss Functions are at the heart of any learning-based algorithm. We convert the learning problem
into an optimization problem, define a loss function and then optimize the algorithm to minimize the loss function.

![loss function loop](/images/playingWithLoss/playingwithlossfunction1.png)*Source: Deep Learning with Python, François Chollet*

Consider a semantic segmentation of *C* objects. This means that there are *C* objects in the image that need to be segmented. We are given a set of images and corresponding annotations for training and developing the algorithm. For simplicity, let us assume that there are *C=3* objects including an ellipse, a rectangle, and a circle. We can use a simple code such as below to generate some masks with three objects.

```python
from skimage.draw import ellipse,polygon,circle
def genEllipse(r,c,h,w,max_rotate=90):
    r_radius=h/6
    c_radius=w/6
    rot_angle=np.random.randint(max_rotate)
    img = np.zeros((h, w), dtype=np.uint8)
    rr, cc = ellipse(r, c, r_radius, c_radius,  rotation=np.deg2rad(rot_angle))
    img[rr, cc] = 1
    return img
def genCircle(r,c,h,w):
    r_radius=h/6
    img = np.zeros((h, w), dtype=np.uint8)
    rr, cc = circle(r, c, r_radius)
    img[rr, cc] = 1
    return img
def genPolygon(r,c,h,w):
    r = np.array([r-h/6, r-h/6, r+h/6, r+h/6])
    c = np.array([c-h/6, c+h/6, c+h/6, c-h/6])
    img = np.zeros((h, w), dtype=np.uint8)
    rr, cc = polygon(r, c )
    img[rr, cc] = 1
    return img
def genMasks(N,C,H,W):
    X=np.zeros((N,C,H,W),"uint8")
    for n in range(N):
        m1=genEllipse(H/4,W/4,H,W)
        m2=genPolygon(3*H/4,3*W/4,H,W)
        m3=genCircle(2*H/4,2*W/4,H,W)
        X[n,0]=m1
        X[n,1]=m2
        X[n,2]=m3
    return X
Y_GT=genMasks(nb_batch,C=3,h,w)
```

Typical ground truth masks for the objects would look like below:

![typical ground truth](/images/playingWithLoss/typicalGroundTruth.png)*Typical ground truth for semantic segmentation.*


Also assume that we develop a deep learning model, which predicts the following outputs:

![Typical predictions](/images/playingWithLoss/typical%20Predictions.png)*Typical model predictions*

First, we are going to use the standard loss function for semantic segmentation, i.e., the categorical cross-entropy as written below:

![Standard categorical cross entropy](/images/playingWithLoss/standardLoss.gif)*Standard categorical cross entropy*


Here *C* is the number of objects, *y_i* is the ground truth and *p_i* is the prediction probability per pixel. Also, *y_i* is one if the pixel belongs to class i and zero otherwise. Note that *i=0* corresponds to the background. The loss will be calculated for all pixels in each image and for all images in the batch. The average of all these values will be a single scalar value reported as the loss value. In the case of categorical cross entropy, the ideal loss will be zero!

To be able to easily debug and compare results, we develop two loss functions, one using Numpy as:

```python
import numpy as np
_EPSILON = 1e-7
nb_class=4 # number of objects plus background
def standard_loss_np(y_true, y_pred):
    y_pred=np.asarray(y_pred,"float32")    
    y_pred[y_pred==0]=_EPSILON
    loss=0
    for cls in range(0,nb_class):
        loss+=y_true[:,:,cls]*np.log(y_pred[:,:,cls])
    return -loss
```


And its equivalent using tensor functions of the [Keras backends](https://keras.io/backend/) as:

```python
from keras import backend as K
_EPSILON = K.epsilon()
nb_class=4 # number of objects plus background
def standard_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    loss=0
    for cls in range(0,nb_class):
        loss+=y_true[:,:,cls]*K.log(y_pred[:,:,cls])
    return -loss
```

As you can see, there is not much difference between the two loss functions except using the backend versus numpy. If we try 8 random annotations and predictions, we obtain *loss_numpy=0.256108245968* and *loss_tensor=0.256108*, from the numpy and the tensor functions, respectively. Practically, the same values!

## Custom Loss Function
Now we are going to develop our own custom loss function. This customization may be needed due to issues in the quality of data and annotations. Let us see a concrete example.

In our case study, let us assume that for some reason there is missing ground truth. For instance, in the below figure there is no ground truth for object 3 (circle) while the deep learning model provides a prediction.

![Ground truth is missing for the circle.](/master/images/playingWithLoss/groundTruthMissing.png)*Ground truth is missing for the circle.*


![Prediction output](/images/playingWithLoss/predictionOutputs2.png)*Prediction output*


In another example, the ground truth is missing for the first object (ellipse):

![Ground truth is missing for the ellipse.](/images/playingWithLoss/GroundTruthMissingEclipse.png)*Ground truth is missing for the ellipse.*


![Prediction outputs.](/master/images/playingWithLoss/predictionOutputs2.png)*Prediction outputs.*

In these scenarios, if we still use the standard loss functions, we may be penalizing the segmentation model incorrectly. The reason is that the pixels belong to the missing ground truth will be considered as the background and multiplied by *-log(p_i)*, where *p_i* is the small prediction probability and as a result *-log(p_i)* is going to be a large number. Note that this is based on our assumption that there should be a ground truth but for whatever reason annotators missed it.


Again, if we try 8 annotations and predictions this time with two random missing annotations, the standard *loss value= 0.493853!* Clearly, this shows a higher loss value compared to when all the ground truths were available.

One easy solution would be to remove the images with missing ground truth. This means that if even one object out of C object has a missing ground truth we have to remove that image from the training data. However, that means less data for training!

Instead, we may be able to develop a smart loss function that avoids such penalization in case of missing ground truth. In this case, we write the loss function as:

![Customized categorical cross entropy](/images/playingWithLoss/customizedCatCrossEntr.gif)*Customized categorical cross entropy*

where *w_i* is the smart weight. If *w_i=1* it will be the same as the standard loss function. We know that if the ground truth is missing for an object, that means that it is assigned as the background. As such if we set *w_0=0* for those pixels that are detected as the object without the ground truth, we will remove any contribution of the background in the loss value. In other words, the custom loss function can be written as below:

![Custom loss function.](/images/playingWithLoss/CustomLossFunc.gif)*Custom loss function.*

To this end, we consider two conditions. First, we find images with missing ground truth. This is possible using:

```python
K.any(y_true,axis=1,keepdims=True)
```

Next, we find the predicted classes per pixel for all images using:

```python
pred=K.argmax(y_pred,axis=-1)
```

Then, we check if the predicted output is, in fact, equal to the missing object. This is also possible using:

```python
K.equal(pred,cls)
```


Note, in the actual implementation, we use:

```python
K.not_equal(pred,cls)
```

since we want both conditions to be False so that the logical-OR is False.

If these two conditions are satisfied we set the background weight equal to zero. This will guarantee that if an object has missing ground truth (in fact mistakenly labeled as background), then the contribution of background in the loss function is zero. The final custom loss function is here:

```python
from keras import backend as K
_EPSILON = K.epsilon()
nb_class=4 # number of objects plus the background
def custom_loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    
    # find predictions
    pred=K.argmax(y_pred,axis=-1)
    # find missing annotations per class
    y_trueAny=K.any(y_true,axis=1,keepdims=True)
    #print "missing annotations",K.eval(y_trueAny)
    
    backgroundWeights=1
    for cls in range(0,nb_class):
        # we use repeat to get the same size as other tensors
                 y_trueAnyRepeat=K.repeat_elements(y_trueAny[:,:,cls],nb_sample,axis=1)
        #print "repeat shape",K.eval(K.shape(y_trueAnyRepeat))
        
        # check for two conditions
        # 1- annotation missing
        # 2- prediction is equal to missing class/object
        backgroundWeights*=K.not_equal(pred,cls)+y_trueAnyRepeat
        #print "background weights shape",K.eval(K.shape(backgroundWeights)),K.eval(K.shape(pred))
        #print "sum of background weights",cls,K.eval(K.sum(backgroundWeig0.191179hts))
    
    # loss for background
    loss=backgroundWeights*y_true[:,:,0]*K.log(y_pred[:,:,0])
    for cls in range(1,nb_class):
        loss+=y_true[:,:,cls]*K.log(y_pred[:,:,cls])
        
    return -loss
    
If we calculate the loss for 8 annotations and two random missing objects we will get custom_loss= 0.191179. This shows that we do not penalize the AI model for providing a correct output just because the ground truth does not exist. In practice, this technique will lead to better overall performance for the objects with missing ground truth.

## Summary
We can always use the standard loss function and they work fine for most cases. However, if you encounter special cases and would like better performance, you can customize the loss function based on your needs.
```

