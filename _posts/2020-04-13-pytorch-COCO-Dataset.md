# PyTorch torchvision COCO Dataset

The PyTorch torchvision package has multiple popular built-in datasets. To see the list of PyTorch built-in datasets, visit the following [link](https://pytorch.org/docs/stable/torchvision/index.html). In this post, we will show you how to create a PyTorch dataset from COCO 2017.

TOC {:toc}

## Downloading COCO Dataset
COCO is a large-scale object detection, segmentation, and captioning dataset. You can find more details about it here. COCO 2017 has over 118K training sample and 5000 validation samples. PyTorch torchvision does not automatically download the COCO dataset. Thus, we need to first download the dataset. Follow the following steps to download the COCO 2017 dataset.

- From a terminal, install pyprotocols:
```
$ git clone https://github.com/pdollar/coco/
$ cd coco/PythonAPI
$ make
$ python setup.py install
$ cd ../..
$ rm -r coco
```

- Install aria2c for faster download:
```
$ sudo apt install aria2c
```

- Download COCO 2017:
```
$ aria2c -x 10 -j 10 http://images.cocodataset.org/zips/train2017.zip
$ aria2c -x 10 -j 10 http://images.cocodataset.org/zips/val2017.zip
$ aria2c -x 10 -j 10 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip *.zip
$ rm *.zip
```

After downloading and unzipping, you will see three folders:
- train2017: training dataset containing 118287 jpg images
- val2017: validation dataset containing 5000 jpg images
- annotations: contains six json files

Create a folder named data and copy the COCO dataset into the folder.

## Create PyTorch Dataset
Now, we can create a PyTorch dataset for COCO. Here we are interested in COCO detection. Therefore, we will use CocoDetection class from torchvision.datasets.


To create a PyTorch dataset for the training data, follow the following steps.
- Import the package:

```
import torchvision.datasets as dset
```

- Define the path to the training and annotation data:
```
path2data="./data/train2017"
path2json="./data/annotations/instances_train2017.json"﻿﻿﻿
```

- Create an object of CocoDetection class for the training dataset:

```
coco_train = dset.CocoDetection(root = path2data,annFile = path2json)
```

- Let us get the number of samples in this coco_train:

```
print('Number of samples: ', len(coco_train))
Number of samples:  118287
```

- Let us look at one sample:
```
img, target-coco_train[0]
print (img.size)
(640, 480)
```

![sample image](/images/cocosample.jpg)

You can follow the same process to create the validation dataset.




