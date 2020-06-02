# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import pandas as pd
import numpy as np
import random
import gc
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  collections import Counter
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import  OneHotEncoder

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


# %%
seq_df = pd.read_pickle(f'{preprocess_path}/ad_id_s64_total_seq.pkl')
print(seq_df)
seq_df = seq_df[seq_df.user_id < 1000000]


# %%
label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')


# %%
total_df = seq_df.merge(label_df,on='user_id',how='left')


# %%
L = 64
emb_model = Word2Vec.load(f'model/ad_id_emb.model_{L}')
print(emb_model)
import numpy as np

vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语

word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
word_vector = {} # 初始化`[word : vector]`字典

# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
embedding_matrix = np.zeros((len(vocab_list) + 1, emb_model.vector_size))

for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1 # 词语：索引
    word_vector[word] = emb_model.wv[word] # 词语：词向量
    embedding_matrix[i + 1] = emb_model.wv[word]  # 词向量矩阵

print(embedding_matrix.shape)


# %%
result=[]
hit=0
miss=0
for row in tqdm(total_df[['user_id','ad_id_seq']].values,total=len(total_df)):
    try:
        result.append([row[0],[word_index[i]  for i in row[-1]]])
        hit+=1
    except Exception as e:
        miss+=1
print(f'hit:{hit}, miss:{miss}')


# %%
int_seq_df  = pd.DataFrame(result,columns=['user_id','ad_id_int_seq'])
print(int_seq_df)


# %%
train_df  = int_seq_df[int_seq_df.user_id <=720000]
valid_df = int_seq_df[int_seq_df.user_id > 720000]

train_df = train_df.merge(label_df,on='user_id',how='left')
train_df['age'] =train_df['age'] -1

valid_df = valid_df.merge(label_df,on='user_id',how='left')
valid_df['age'] =valid_df['age'] -1


train_x = np.array(train_df[['ad_id_int_seq']].values[:,0])
train_y = train_df[['age']].values

valid_x = np.array(valid_df[['ad_id_int_seq']].values[:,0])
valid_y = valid_df[['age']].values

before_one_hot =  train_y.reshape([-1,1])
enc = OneHotEncoder()
enc.fit(before_one_hot)

one_hoted_train_y  = enc.transform(before_one_hot).toarray()
print(one_hoted_train_y.shape)

before_one_hot =  valid_y.reshape([-1,1])
print(before_one_hot)
enc = OneHotEncoder()
enc.fit(before_one_hot)

one_hoted_valid_y  = enc.transform(before_one_hot).toarray()
print(one_hoted_valid_y.shape)

print(train_x)
print(len(train_x))
maxlen = 1000
train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=maxlen)
valid_x = keras.preprocessing.sequence.pad_sequences(valid_x, maxlen=maxlen)
print(train_x)


# %%

embedding_layer = Embedding(
    len(vocab_list) +1,
    emb_model.vector_size,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)


# %%

inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = embedding_layer(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# %%
print(train_x.shape)
print(one_hoted_train_y)
print(valid_x.shape)
print(one_hoted_valid_y)

model.fit(train_x,one_hoted_train_y, validation_data=(valid_x,one_hoted_valid_y), epochs=3)


# %%



