#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen
# 文本-情感 数据的处理与训练

import numpy as np
import pandas as pd
from gensim import corpora
from fnn_diy import fnn_run, fnn_train

# 数据的导入与预处理
train_data = pd.read_csv("sentiment-analysis-on-movie-reviews/train.tsv",
                         sep='\t', encoding='ISO-8859-1')
phrase_id = train_data.loc[:, 'PhraseId'].values
phrase = train_data.loc[:, 'Phrase'].values
sentiment = train_data.loc[:, 'Sentiment'].values

# 由原始表格数据生成词典与词向量
train_texts = [[word for word in sentence.lower().split()] for sentence in
               phrase]  # lower()使大写变小写
dictionary = corpora.Dictionary(train_texts)  # 词典的生成,可以根据列表或列表的列表
# gensim中的dictionary实际上是一个单词到id的唯一映射,是一种词典,id从0开始计算
corpus = [dictionary.doc2bow(text) for text in train_texts]  # 稀疏one-hot向量形式
dict_len = len(dictionary)  # 词典中词的总数

# 由稀疏的bow向量生成稠密的文本特征向量
word_feature = np.zeros((dict_len, len(phrase)), dtype='uint8')  # unit8降低内存
for i in range(len(corpus)):
    for bow in corpus[i]:
        word_feature[bow[0], i] = bow[1]

# 情感标签特征向量的生成(one-hot形式)
sentiment_vec = np.zeros((5, len(sentiment)))
for i in range(len(sentiment)):
    sentiment_vec[sentiment[i], i] = 1

# 3层fnn训练,划分训练集与2万测试集,计算准确率
set_layer = 3
set_neurons = [100, 30, 5]
weight_t, bias_t = fnn_train(word_feature[:, 0:136060],
                             sentiment_vec[:, 0:136060],
                             set_layer, set_neurons, iterations=50)

# 测试集上测试
test_results = fnn_run(word_feature[:, 136060:156060], set_layer,
                       weight_t, bias_t)[1][-1]
test_sentiment = np.argmax(test_results, axis=0)
accuracy = (test_sentiment == sentiment[136060:156060]).mean()
print("accuracy = %s" % accuracy)

# 在训练数据上测试
self_test_results = fnn_run(word_feature[:, 116060:136060], set_layer,
                            weight_t, bias_t)[1][-1]
self_test_sentiment = np.argmax(self_test_results, axis=0)
self_accuracy = (self_test_sentiment == sentiment[116060:136060]).mean()
print("self_accuracy = %s" % self_accuracy)

# 导入预测集并进行数据处理
test_data = pd.read_csv("sentiment-analysis-on-movie-reviews/test.tsv",
                        sep='\t', encoding='ISO-8859-1')
predict_phrase = test_data.loc[:, 'Phrase'].values
predict_phrase_id = test_data.loc[:, 'PhraseId'].values
predict_texts = [[word for word in sentence.lower().split()] for sentence in
                 predict_phrase]
predict_corpus = [dictionary.doc2bow(text) for text in predict_texts]

# 由稀疏的bow向量生成稠密的文本特征向量
predict_feature = np.zeros((dict_len, len(predict_phrase)), dtype='uint8')
for i in range(len(predict_corpus)):
    for bow in predict_corpus[i]:
        predict_feature[bow[0], i] = bow[1]

# 情感预测并导入
predict_results = fnn_run(predict_feature, set_layer, weight_t, bias_t)[1][-1]
predict_sentiment = np.argmax(predict_results, axis=0)
predict_dataFrame = pd.DataFrame({'PhraseId': predict_phrase_id,
                                  'Sentiment': predict_sentiment})
predict_dataFrame.to_csv('predict01.csv', index=False, sep=',')
