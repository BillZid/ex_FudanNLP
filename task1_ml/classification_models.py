#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

"""包含了机器学习中各个线性分类模型的实现.

有softmax回归,感知器等模型.
"""

import numpy as np


def softmax(x):
    """softmax函数升级版

    可以直接作用于向量,作用于矩阵时,相当于对矩阵各列向量进行softmax,再拼接
    """
    vector_1 = np.ones((x.shape[0], x.shape[0]))
    z = np.exp(x) / np.matmul(vector_1, np.exp(x))  # 此处是矩阵按元素相除
    return z


def softmax_train(features, labels, alpha=0.5, iterations=200):
    """
    这是一个softmax回归模型的训练.

    此算法依赖numpy库,训练参数时使用梯度下降法,预测的属于类别c的条件概率为
    exp(w_c^T*x)/sum(exp(w_c^T*x)).

    Args:
        features:样本训练集中的特征向量x组成的矩阵,每一列代表一组
        labels:样本训练集中的标签y组成的矩阵,每一列代表一组
        alpha:迭代更新权重向量时采用的学习率,取值范围0-1
        iterations:训练时的迭代次数

    Returns:
        输出softmax函数中以w为列构成的权重矩阵W.
    """
    aug_vector = np.ones((1, features.shape[1]), dtype='uint8')  # 加上偏置
    # 使特征写成增广形式
    features = np.concatenate((features, aug_vector), axis=0)
    size = features.shape[1]  # N,代表样本数目
    weight = np.zeros((features.shape[0], labels.shape[0]))  # W,代表权重矩阵
    for t in range(iterations):
        w_temp = np.zeros((features.shape[0], labels.shape[0]))
        for n in range(size):
            x_n = features[:, n]
            y_n = labels[:, n]
            y_w_n = softmax(np.dot(weight.T, x_n))
            w_temp = w_temp + alpha / size * np.outer(x_n, (y_n - y_w_n).T)
        weight = weight + w_temp
    return weight


def softmax_calculate(weight, features):
    """
    这是一个直接计算出softmax预测值(预测标签分布)的函数.

    Args:
        weight: 所得到的softmax权重矩阵
        features: 用来预测的特征x,为矩阵形式,每一列代表一组特征

    Returns:
        得到一个代表预测出的标签的概率分布的矩阵,每一列为一组预测
    """
    aug_vector = np.ones((1, features.shape[1]))  # 加上偏置,使特征写成增广形式
    features = np.concatenate((features, aug_vector), axis=0)
    labels = softmax(np.dot(weight.T, features))
    return labels
