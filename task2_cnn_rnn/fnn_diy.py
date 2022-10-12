#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

"""尝试实现前馈神经网络模型的搭建(FNN)

"""

from typing import Tuple, List, Any

import numpy as np
from numpy import ndarray, float64


def softmax(x: ndarray) -> ndarray:
    """softmax函数"""
    vector_1 = np.ones((x.shape[0], 1))
    z = np.exp(x) / np.matmul(vector_1.T, np.exp(x))
    return z


def logistic(x) -> float64:
    """logistic函数"""
    z = 1 / 1 + np.exp(-x)
    return z


def tanh(x) -> float64:
    """Tanh函数"""
    z = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return z


def bgd(theta, gradient, learning_rate=0.5, iterations=50):
    """(批量)梯度下降法,训练标量或向量均可

    Args:
        theta ():参数初始值
        gradient ():梯度函数
        learning_rate ():学习率
        iterations (): 迭代次数
    """
    for i in range(iterations):
        theta = theta - learning_rate * gradient(theta)
    return theta


def fnn_run(features, no_layer, weight, bias,
            activation=logistic) -> Tuple[List[Any], List[Any]]:
    """前馈神经网络FNN的运行(可以对多个向量求解)

    Args:
        features (ndarray):数据集中的特征向量x组成的矩阵,每一列代表一组
        no_layer (int):神经网络的层数
        weight (list):各层间的权重矩阵
        bias (list):各层间的偏置
        activation (function):各层的激活函数

    Returns:
        各层的净输入与输出(可以对多个向量求解)

        """
    a = [features]
    z = []
    vector_1 = features.shape[1]  # 向量(样本)的个数
    for l in range(no_layer - 1):
        z.append(weight[l] * a[-1] + np.out(bias[l], vector_1))
        a.append(activation(z[-1]))
    z.append(weight[no_layer] * a[-1] + bias[no_layer])
    a.append(softmax(z[-1]))
    return z, a


def back_propagation(features, labels, no_layer, weight, bias,
                     activation = logistic) -> list:
    """前向传播算法计算每一层的误差项

    Args:
        features (ndarray):数据集中的特征向量x组成的矩阵,每一列代表一组
        labels (ndarray):训练集中的标签y组成的矩阵,每一列代表一组
        no_layer (int):神经网络的层数
        weight (list):各层间的权重矩阵
        bias (list):各层间的偏置
        activation (function):各层的激活函数

    Returns:
        神经网络每一层的误差项

    """
    z, a = fnn_train(features, no_layer, weight, bias, activation)
    delta = []  # 误差项
    return 0


def fnn_train(features, labels, no_layer, no_neurons,
              activation=logistic) -> Tuple[List[Any], List[Any]]:
    """训练前馈神经网络FNN

    Args:
        features (ndarray):训练集中的特征向量x组成的矩阵,每一列代表一组
        labels (ndarray):训练集中的标签y组成的矩阵,每一列代表一组
        no_layer (int):神经网络的层数
        no_neurons (list):各层的神经元个数
        activation (function):各层的激活函数

    Returns:
        各层的权重矩阵以及偏置

    """
    weight = []
    bias = []
    return weight, bias
