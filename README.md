# SVR

Code for paper "Saliency-aware Stereoscopic Video Retargeting"


![alt text](https://github.com/z65451/SVR/blob/main/model.jpg)

![alt text](https://github.com/z65451/SVR/blob/main/result.png)

# Training

First install the requirements:

```
pip3 install requirements
```


## Download the datasets
First fownload the KITTI stereo video 2015 and 2012 datasets from the following link:

https://www.cvlibs.net/datasets/kitti/eval_stereo.php

Unzip and put the datasets in the "datasets/" directory.

## Download the weights of CoSD [1]

Download the weights of the saliency detection method from the following like:

https://github.com/suyukun666/UFO

Put them in the "models/" directory.

For calculating the perceptual loss [2], clone the following repository and put it in the "PerceptualSimilarity/" directory:

https://github.com/SteffenCzolbe/PerceptualSimilarity

## Start training

Run the following command to start the training:

```
sudo python3 main.py --train_or_test train
```

The trained model will be saved in the "models/" directory.


# Testing
Download the pre-trained model from th following link as put it in "models/" directory:

https://drive.google.com/file/d/11mlx1PRh-oFzOTLABAkySk7_DGLvEmyw/view?usp=share_link

Run the following command to start testing:

```
sudo python3 main.py --train_or_test test
```


# References

[1] Yukun Su, Jingliang Deng, Ruizhou Sun, Guosheng Lin,
and Qingyao Wu. A unified transformer framework for
group-based segmentation: Co-segmentation, co-saliency
detection and video salient object detection. arXiv preprint
arXiv:2203.04708, 2022. 3

[2] Czolbe, Steffen, et al. "A loss function for generative neural networks based on watson’s perceptual model." Advances in Neural Information Processing Systems 33 (2020): 2051-2061.
