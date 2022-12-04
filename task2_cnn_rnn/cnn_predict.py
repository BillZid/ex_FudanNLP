#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import corpora

from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=100)

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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv1d(100, 20, 2)
        self.conv2 = nn.Conv1d(100, 20, 3)
        self.conv3 = nn.Conv1d(100, 20, 4)
        self.fc1 = nn.Linear(60, 5)

    def forward(self, x):
        # Max pooling
        x1 = F.max_pool1d(F.relu(self.conv1(x)), 51)
        x2 = F.max_pool1d(F.relu(self.conv2(x)), 50)
        x3 = F.max_pool1d(F.relu(self.conv2(x)), 49)
        x = torch.flatten(torch.cat((x1, x2, x3), 0))
        x = F.softmax(self.fc1(x), 0)
        return x


net = Net()
net_state_dict = torch.load('./glove_cnn_net.pth')
net.load_state_dict(net_state_dict)

correct = 0
total = 0
with torch.no_grad():
    for i in range(136060, 156060):
        word_embed = glove.get_vecs_by_tokens(train_texts[i], True)
        zero_embed = torch.zeros(52 - word_embed.size()[0], 100)
        inputs = torch.cat((word_embed, zero_embed), 0).T
        labels = sentiment[i]
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test data: %d %%' % (
        100 * correct / total))

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
correct = 0
total = 0
predict_sentiment = []
predict_texts[1390] = ['0']
with torch.no_grad():
    for i in range(predict_feature.shape[1]):
        word_embed = glove.get_vecs_by_tokens(predict_texts[i], True)
        if word_embed.size()[0] > 52:
            word_embed = word_embed[0:52]
        zero_embed = torch.zeros(52 - word_embed.size()[0], 100)
        inputs = torch.cat((word_embed, zero_embed), 0).T
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 0)
        predict_sentiment.append(predicted.numpy())

predict_dataFrame = pd.DataFrame({'PhraseId': predict_phrase_id,
                                  'Sentiment': predict_sentiment})
predict_dataFrame.to_csv('predict03.csv', index=False, sep=',')
