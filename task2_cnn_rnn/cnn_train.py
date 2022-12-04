#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Wenzhen

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim import corpora

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
        self.conv1 = nn.Conv1d(1, 1, 3)
        self.fc1 = nn.Linear(8269, 5)

    def forward(self, x):
        # Max pooling
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.softmax(self.fc1(x), 1)
        return x


net = Net()
# print(net)

# params = list(net.parameters())
# print(len(params))
# for i in range(12):
#   print(params[i].size())

# input = torch.from_numpy(word_feature[:, 0].T, ).reshape(1, 16540).float()
# print(input)
# out = net(input)
# print(out)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
# print(device)
net.to(device)

start = time.time()
for epoch in range(1):

    running_loss = 0.0
    for i in range(136060):
        # get the inputs
        inputs = torch.from_numpy(word_feature[:, i].T, ).reshape(1,
                                                                  16540).float()
        labels = torch.from_numpy(sentiment_vec[:, i].T, ).reshape(1, 5).float()
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10000 == 9999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10000))
            running_loss = 0.0

print('Finished Training')
end = time.time()
print(f'cost time: {end - start} seconds')
# cost time: 1727.3624396324158 seconds

PATH = './cifar_net1.pth'
torch.save(net.state_dict(), PATH)

