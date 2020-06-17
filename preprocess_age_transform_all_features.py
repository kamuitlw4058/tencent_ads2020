# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import os
import pandas as pd
import numpy as np
import random
import gc
import math
import json

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


def get_emb_mat(f,L=64,flag='clk'):
    emb_dic_path = f'model/{f}_{flag}_emb_dict.json'
    emb_mat_path = f'model/{f}_{flag}_emb_mat.npy'
    if os.path.exists(emb_dic_path) and os.path.exists(emb_mat_path):
        with open(emb_dic_path,'r') as load_f:
            word_index = json.load(load_f)
        embedding_mat = np.load(emb_mat_path)
    else:
        emb_model = Word2Vec.load(f'model/{f}_{flag}_emb.model_{L}')
        print(emb_model)

        vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语
        word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
        word_vector = {} # 初始化`[word : vector]`字典

        embedding_mat = np.zeros((len(vocab_list) + 1, emb_model.vector_size))

        for i in range(len(vocab_list)):
            # print(i)
            word = vocab_list[i]  # 每个词语
            word_index[word] = i + 1 # 词语：索引
            word_vector[word] = emb_model.wv[word] # 词语：词向量
            embedding_mat[i + 1] = emb_model.wv[word]  # 词向量矩阵

        with open(f'model/{f}_{flag}_emb_dict.json','w') as f:
            f.write(json.dumps(word_index))

        np.save(f'model/{f}_{flag}_emb_mat.npy',embedding_mat)
        print(embedding_mat.shape)
    return word_index,embedding_mat
    
    
def get_seq_data(seq_df,f,maxlen,L=64,flag='clk',dataset='train'):
    word_index,embedding_mat = get_emb_mat(f,L=L,flag=flag)
    seq_values_path =  f'preprocess/{dataset}_{f}_{flag}_maxlen{maxlen}_int_seq.npy'
    if os.path.exists(seq_values_path):
        seq_values = np.load(seq_values_path)
    else:
        result=[]
        hit=0
        miss=0
        for row in tqdm(seq_df[['user_id',f'{f}_seq']].values,total=len(seq_df)):
            try:
                result.append([row[0],[word_index[i]  for i in row[-1]]])
                hit+=1
            except Exception as e:
                miss+=1
        print(f'hit:{hit}, miss:{miss}')

        int_seq_df  = pd.DataFrame(result,columns=['user_id',f'{f}_int_seq'])

        seq_values = np.array(train_df[[f'{f}_int_seq']].values[:,0])
        seq_values = keras.preprocessing.sequence.pad_sequences(seq_values, maxlen=maxlen)
        print(f"end pad seq ")
        np.save(seq_values_path,seq_values)
        
    return  word_index,embedding_mat,seq_values
    

maxlen = 150

train_x_list = []
valid_x_list = []
vocab_size_list = []


for i in ['time', 'creative_id', 'ad_id','product_id','advertiser_id','product_category','industry']:
    print(f'start {i}...')
    seq_df = pd.read_pickle(f'{preprocess_path}/{f}_s64_total_seq.pkl').sort_values(by='user_id')
    seq_df = seq_df[seq_df.user_id < 1000000]
    word_index,embedding_mat,seq_values = get_seq_data(seq_df,i,maxlen)



    


