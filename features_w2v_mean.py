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
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'



def seq2mean(f,topN=1,size=64,pivot='user_id',flag='train'):
    print("start to seq 2 mean " + f)
    path = f'{preprocess_path}/{f}_s{size}_total_seq.pkl'
    user_df = pd.read_pickle(path)
    user_df = user_df[user_df.user_id < 1000000]    
    print(user_df)

    size = 64
    output_df   = w2v(user_df,pivot,f,flag,size,f'/data/workspace/kimi/tencent_ads/2020/kimi/model/{f}_emb.model',topN=topN)
    print(output_df)
    output_topn = f'{f}_top{topN}_s{size}'
    output_topn_path = f'{preprocess_path}/{output_topn}'
    output_df.to_pickle(output_topn_path)




def forfor(a): 
    return [item for sublist in a for item in sublist] 

def w2v(log,pivot,f,flag,L,model_path,topN=3):
    #训练Word2Vec模型
    model = Word2Vec.load(model_path+ f'_{L}')
    print(model)
    
    is_first_user = True
    result=[]
    print('outputing...')
    for row in tqdm(log[['user_id',f'{f}_seq']].values,total=len(log)):
        user_sentence = None
        c = Counter()
        for w in row[1]:
            c.update([w])
            try:
                emb_vec =  model.wv[w]
            except Exception as e:
                emb_vec = [0  for i in range(L)]
            
            if user_sentence is None:
                user_sentence = np.array(emb_vec)
            else:
                user_sentence = user_sentence + np.array(emb_vec)
                
        if user_sentence is None:
            new_list = [0  for i in range(L)]
            user_sentence = np.array(new_list)
        user_sentence = user_sentence / len(row[1])
        key_counts =[]
        for k,v in  c.items():
            key_counts.append(v)
        key_counts = np.array(key_counts)
        
        top_list = c.most_common(topN)
        top_list_len = len(top_list)
        if top_list_len < topN:
            rlen = topN- top_list_len
            top_list = top_list + ["-2" for i in range(rlen)]
        top_ret = []
        for t in top_list:
            try:
                top_vec_count = t[1]
                top_vec =  model.wv[t[0]]
            except Exception as e:
                top_vec = np.array([0  for i in range(L)])
            top_ret = top_ret + top_vec.flatten().tolist() + [top_vec_count]
            
        if len(top_ret) != L * topN + topN:
            print(f"len error!{len(top_ret)} need {L * topN + topN}")
        data= [row[0]] + user_sentence.flatten().tolist() + top_ret + [np.mean(key_counts),np.std(key_counts),np.min(key_counts)]
        result.append(data)
    cols = ['user_id'] + [f'{f}_{i}'  for i in range(L)]  +forfor([[f'{f}_top{i}_{j}'  for j in range(L + 1)]  for i in range(topN)])  + [f'{f}_mean',f'{f}_std',f'{f}_min']  
    ret_df = pd.DataFrame(result,columns=cols)
    #保存文件
    return ret_df 


# %%
seq2mean('industry')
gc.collect()
seq2mean('advertiser_id')
gc.collect()
seq2mean('product_id')
gc.collect()
seq2mean('ad_id')
gc.collect()
seq2mean('creative_id')
gc.collect()




