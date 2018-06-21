"""
构建CIFAR-10网络
本文件函数功能呢摘要：

获得用于训练的图像和标签，如果你想要运行评估，可以使用inputs（）
images, labels = distorted_inputs()

计算前向过程，进行预测
predictions = inference(inputs)

计算预测值相对于标签值的损失
loss = loss(predictions, labels)

创建一个graph，对loss进行训练优化
train_op = train(loss, global_step)
"""

import os
import re
import sys
import tarfile

import tensorflow as tf
