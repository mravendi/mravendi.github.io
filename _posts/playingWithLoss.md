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

![typical ground truth](/images/playingWithLoss/typicalGroundTruth.png)*Typical ground truth for objects.*


