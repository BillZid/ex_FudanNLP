#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

"""尝试实现前馈神经网络模型的搭建(FNN)

"""

from typing import Tuple, List, Any, Union

import numpy as np
from numpy import ndarray, float64


def softmax(x: ndarray) -> ndarray:
    """softmax函数升级版

    可以直接作用于向量,作用于矩阵时,相当于对矩阵各列向量进行softmax,再拼接
    """
    vector_1 = np.ones((x.shape[0], x.shape[0]))
    z = np.exp(x) / np.matmul(vector_1, np.exp(x))  # 此处是矩阵按元素相除
    return z


def logistic(x):
    """logistic函数"""
    z = 1 / 1 + np.exp(-x)
    return z


def logistic_d(x):
    """logistic函数对标量的导数"""
    z = logistic(x) * (1 - logistic(x))
    return z


def tanh(x):
    """Tanh函数"""
    z = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return z


def bgd(theta, gradient, learning_rate=0.5, regular_term=0, iterations=50):
    """(批量)梯度下降法,训练标量或向量均可

    Args:
        regular_term (): 正则项的系数
        theta ():参数初始值
        gradient ():梯度函数
        learning_rate ():学习率
        iterations (): 迭代次数
    """
    for i in range(iterations):
        theta = theta - learning_rate * (gradient(theta) + regular_term * theta)
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
    a = [features]  # 把样本特征作为第0层输出: a0
    z = []
    # 为了对多个样本向量组成的矩阵直接计算
    vector_1 = np.ones((1, features.shape[1]))  # 向量(样本)的个数
    # 信息传播
    for i in range(no_layer - 1):
        z.append(np.dot(weight[i], a[-1]) + np.outer(bias[i], vector_1))
        # dot是矩阵乘法,outer是向量外积,为a*(b.T)
        a.append(activation(z[-1]))
    # 最后一层softmax,默认多分类对应最后一层输出层神经元数量
    # 注意此处直接用softmax而不是logistic,故二分类时最后一层神经元数量应设为2
    z.append(np.dot(weight[no_layer - 1], a[-1]) + np.outer(bias[no_layer - 1],
                                                            vector_1))
    a.append(softmax(z[-1]))
    del a[0]  # 删去不需要的第0层输出 a0
    return z, a


def para_init(no_layer, no_neurons, no_features):
    """神经网络权重与偏置的初始化(Xavier原则)

    Args:
        no_layer (int):神经网络的层数
        no_neurons (list):各层的神经元个数
        no_features (int):特征的维数

    Returns:
        初始权重 (list),初始偏置 (list)

    """
    no_neurons.insert(0, no_features)  # 插入首位
    weight = []
    bias = []
    for i in range(no_layer):
        weight.append(np.random.randn(no_neurons[i + 1], no_neurons[i]) /
                      np.sqrt(no_neurons[i]))  # S型激活函数用Xavier原则确定初始权重
        bias.append(np.zeros(no_neurons[i + 1]))  # 直接取0
    return weight, bias


def back_propagation(features, labels, no_layer, weight, bias,
                     activation=logistic) -> Tuple[List, List]:
    """前向传播算法计算损失函数关于每个参数的导数

    Args:
        features (ndarray):数据集中的特征向量x组成的矩阵,每一列代表一组
        labels (ndarray):训练集中的标签y组成的矩阵,每一列代表一组
        no_layer (int):神经网络的层数
        weight (list):各层间的权重矩阵
        bias (list):各层间的偏置
        activation (function):各层的激活函数

    Returns:
        损失函数关于每个参数的导数

    """
    z, a = fnn_run(features, no_layer, weight, bias, activation)
    no_features = features.shape[1]  # 样本的数量
    gradient_w = []  # 按神经层的编号索引
    gradient_b = []
    gradient_w_no = 0  # 按样本编号索引,此处为初始化
    gradient_b_no = 0
    for layer in range(no_layer -1):
        for no_f in range(no_features):  # 第几个样本
            delta = [(labels - a[-1])[:, no_f]]  # 最后一层误差项为 yn - yn^
            for i in range(no_layer - 2, -1, -1):  # 从L-1层(索引为L-2)往前迭代传播
                delta.append(logistic_d(z[i][:, no_f]) * np.dot(weight[i + 1].T,
                                                                delta[-1]))
            delta.reverse()
            if no_f == 0:
                gradient_w_no = np.outer(delta[layer + 1], a[layer][:, no_f])
                gradient_b_no = delta[layer]
            else:
                gradient_w_no += np.outer(delta[layer + 1], a[layer][:, no_f])
                gradient_b_no += delta[layer]
        gradient_w.append(gradient_w_no / no_features)
        gradient_b.append(gradient_b_no / no_features)
    return gradient_w, gradient_b


def fnn_train(features, labels, no_layer, no_neurons,
              learning_rate=0.5, regular_term=0.2, iterations=50,
              activation=logistic) -> Tuple[List[Any], List[Any]]:
    """训练前馈神经网络FNN

    Args:
        features (ndarray):训练集中的特征向量x组成的矩阵,每一列代表一组
        labels (ndarray):训练集中的标签y组成的矩阵,每一列代表一组
        no_layer (int):神经网络的层数
        no_neurons (list):各层的神经元个数
        regular_term (float): 正则项的系数
        learning_rate (float):学习率
        iterations (int): 迭代次数
        activation (function):各层的激活函数

    Returns:
        各层的权重矩阵以及偏置

    """
    weight, bias = para_init(no_layer, no_neurons, features.shape[0])
    # 随机梯度下降
    for i in range(iterations):
        gradient_w, gradient_b = back_propagation(features, labels, no_layer,
                                                  weight, bias, activation)
        for j in range(no_layer):
            weight[j] = weight[j] - learning_rate * (gradient_w[j] +
                                                     regular_term * weight[j])
            bias[j] = bias[j] - learning_rate * (gradient_b[j])
    return weight, bias


# 代码运行测试
train_features = np.array([[1, 1], [1, 0], [1, 1], [1, 1]])
train_labels = np.array([[1], [0]])
set_layer = 3
set_neurons = [5, 3, 2]
weight, bias = fnn_train(train_features, train_labels, set_layer, set_neurons)
# test_value = fnn_run(test_feature, set_layer, weight, bias)
# test_result = test_value[1][-1]
