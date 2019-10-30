# Depth-Map-Generation

## Introduction
To generate the disparity map given the left and right images, we utilize
learning-based method and our understanding of stereo geometry. By training
an end-to-end model, it can generate the disparity map only with the
information of the both images.

## Pre-processing


### Date type Detection
The relationship between left images and right ones are quite different when solving Real image cases. The most significant part we've observed is the relative position. In real image cases, the perspectives of right images are only little shifted from left images.
As a result, we collect the features of both images, and calculate the corresponding distance. It is evident that the distance of real images is much lower than that of the synthetic ones. To deal with this problem, we set a threshold to classify the real images and fake images, and shift the right images to fit the deep model that training on the dataset where the distance between left and right image is rather higher, at the same time crop the left image.
By the way, more importantly, we use our GC-net model on both the real data and synthetic data, which means that our model does not overfit to training data, and at the same time can generalize to the images in hw4.
