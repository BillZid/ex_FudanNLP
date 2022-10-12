#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

"""包含了机器学习中各个线性分类模型的实现.

有softmax回归,感知器等模型.
"""

import numpy as np


def softmax(sample_features, sample_labels, alpha=0.5, iterations=50):
    """
    这是一个softmax回归模型的算法实现.

    此算法依赖numpy库,训练参数时使用梯度下降法,预测的属于类别c的条件概率为
    exp(w_c^T*x)/sum(exp(w_c^T*x)).

    Args:
        sample_features:样本训练集中的特征向量x组成的矩阵,每一列代表一组
        sample_labels:样本训练集中的标签y组成的矩阵,每一列代表一组
        alpha:迭代更新权重向量时采用的学习率,取值范围0-1
        iterations:训练时的迭代次数

    Returns:
        输出softmax函数中以w为列构成的权重矩阵W.
    """
    sample_size = sample_features.shape[1]  # N,代表样本数目
    dimension = sample_features.shape[0]  # D,代表样本维度
    class_number = sample_labels.shape[0]  # C,代表类别数目
    weight_matrix = np.zeros((dimension, class_number))  # W,代表权重矩阵
    vector_1 = np.ones((class_number, 1))
    for t in range(iterations):
        w_temp = np.zeros((dimension, class_number))
        for n in range(sample_size):
            x_n = sample_features[:, n]
            y_n = sample_labels[:, n]
            y_w_n = np.exp(np.matmul(weight_matrix.T, x_n)) / np.matmul(
                vector_1.T, np.exp(np.matmul(weight_matrix.T, x_n)))
            w_temp = w_temp + alpha * np.matmul(np.array([x_n]).T,
                                                [(y_n - y_w_n)]) / sample_size
        weight_matrix = weight_matrix + w_temp
        return weight_matrix


def softmax_calculate(weight, predict_features):
    """
    这是一个直接计算出softmax预测值(预测标签分布)的函数.

    Args:
        weight: 所得到的softmax权重矩阵
        predict_features: 用来预测的特征x,为矩阵形式,每一列代表一组特征

    Returns:
        得到一个代表预测出的标签的概率分布的矩阵,每一列为一组预测
    """
    vector_1 = np.ones((weight.shape[1], 1))
    labels = np.exp(np.matmul(weight.T, predict_features)) / np.matmul(
        vector_1.T,
        np.exp(
            np.matmul(weight.T,
                      predict_features)))
    return labels
