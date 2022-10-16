#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen
# 文本-情感 数据的处理与训练

from random import randint
import pandas as pd

# 数据的导入与预处理
源数据 = pd.read_csv("sentiment-analysis-on-movie-reviews/train.tsv",
                     sep='\t', encoding='ISO-8859-1')
句子id = 源数据.loc[:, 'PhraseId'].values
句子 = 源数据.loc[:, 'Phrase'].values
情感类型 = 源数据.loc[:, 'Sentiment'].values
情感值统计 = [0, 0, 0, 0, 0]
for 情感 in 情感类型:
    情感值统计[情感] += 1
随机情感类型 = []
for i in range(156060):
    随机数 = randint(1, 5)
    随机情感类型.append(随机数)
准确率 = (随机情感类型 == 情感类型).mean()
测试部分准确率 = (随机情感类型[136060:156060] == 情感类型[136060:156060]).mean()
print("准确率 = %s" % 准确率)
print("测试部分准确率 = %s" % 测试部分准确率)
