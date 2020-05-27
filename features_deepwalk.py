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

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


# %%
total_merged= 'total_merged.pkl'
total_merged_path = f'{preprocess_path}/{total_merged}'
total_merged_df = pd.read_pickle(total_merged_path)
print(total_merged_df)


# %%

def deepwalk(log,f1,f2,flag,L):
    #Deepwalk算法，
    print("deepwalk:",f1,f2)
    #构建图
    dic={}
    for item in tqdm(log[[f1,f2]].values,total=len(log)):
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))]=set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))]=set(['item_'+str(int(item[1]))])
    dic_cont={}
    for key in dic:
        dic[key]=list(dic[key])
        dic_cont[key]=len(dic[key])
    print("creating")
    #构建路径
    path_length=10
    sentences=[]
    length=[]
    for key in tqdm(dic,total=len(dic)):
        sentence=[key]
        while len(sentence)!=path_length:
            key=dic[sentence[-1]][random.randint(0,dic_cont[sentence[-1]]-1)]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%100000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    #训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4,min_count=1,sg=1, workers=10,iter=20)
    print('outputing...')
    model.save(f'deepwalk_{f2}_{L}')
    #输出
    values=set(log[f1].values)
    w2v=[]
    for key in dic:
        if key.startswith('user_'):
            w2v.append([key[5:]] + model[key].flatten().tolist())
    out_df=pd.DataFrame(w2v,columns=['user_id'] + [f1+'_'+ f2+'_'+f1+'_deepwalk_embedding_'+str(L)+'_'+str(i) for i in range(L)])
    print(out_df.head())
    out_df.to_pickle(f'{preprocess_path}/' +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')
    ########################
    values=set(log[f2].values)
    w2v=[]
    for key in dic:
        if key.startswith('item_'):
            w2v.append([key[5:]] +  model[key].flatten().tolist())
    out_df=pd.DataFrame(w2v,columns=['user_id'] + [f1+'_'+ f2+'_'+f2+'_deepwalk_embedding_'+str(L)+'_'+str(i) for i in range(L)])
    print(out_df.head())
    out_df.to_pickle(f'{preprocess_path}/' +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')

# %%
deepwalk(total_merged_df,'user_id','advertiser_id','train',64)
deepwalk(total_merged_df,'user_id','industry','train',64)


# %%



