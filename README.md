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

### histogram equalization

We observed that the given right and left image have different brightness. In contrast, those pairs in training set have similar brightness. As a result, we performed "Histogram Equalization" (OpenCV) to reduce their brightness difference and the resulted disparity maps were greatly improved.

### median blur+guided filter

After histogram equalization, we employ median blur to reduce trivial details and use guided filter to sharpen edges based on input images to keep contours.

## Reference

* End-to-End Learning of Geometry and Context for Deep Stereo Regression. Alex Kendall Hayk Martirosyan Saumitro Dasgupta Peter Henry Ryan Kennedy Abraham Bachrach Adam Bry Skydio Inc ICCV2017.
* Self-Supervised Learning for Stereo Matching with Self-Improving Ability. Yiran Zhong , Yuchao Dai, and Hongdong Li Australian National University, Australian Centre for Robotic Vision, Data61.
