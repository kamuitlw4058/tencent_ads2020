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
total_final = 'total_final.pkl'
total_final_path = f'{preprocess_path}/{total_final}'
total_final_df = pd.read_pickle(total_final_path)
print(total_final_df)


# %%
train_df = total_final_df[total_final_df.user_id <= 720000]
valid_df = total_final_df[total_final_df.user_id > 720000]
valid_df = valid_df[valid_df.user_id < 2000000]
# print(train_df)
# print(valid_df)
topN = 3


# %%

def forfor(a):
    return [item for sublist in a for item in sublist]

def w2v(log,pivot,f,flag,L,model_path,topN=3):
    #训练Word2Vec模型
    model = Word2Vec.load(model_path+ f'_{L}')
    print(model)

    is_first_user = True
    result=[]
    print('w2v mean seq...')
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



featues = [
            ('industry',16,'/data/workspace/kimi/tencent_ads/2020/kimi/industry_emb_model'),
            ('product_id',32,'/data/workspace/kimi/tencent_ads/2020/kimi/product_id_emb_model'),
            ('advertiser_id',32,'/data/workspace/kimi/tencent_ads/2020/kimi/advertiser_id_emb_model'),
          ]
dataset = [
    (train_df,featues,'train'),
    (valid_df,featues,'valid')
 ]

for user_df,featues,flag in dataset:
    for f, size, model_path in featues:
        print(f'start flag:{flag}  f:{f}')
        ret_df =  w2v(user_df,'user_id',f,flag, size,model_path,topN=topN)
        ret_name = f'{f}_top{topN}_l{size}_{flag}.pkl'
        ret_path = f'{preprocess_path}/{ret_name}'
        ret_df.to_pickle(ret_path)
        del ret_df




def get_product_category(log):




# # %%
# product_id_size = 32
# product_id_df   = w2v(train_df,'user_id','product_id','train',product_id_size,'/data/workspace/kimi/tencent_ads/2020/kimi/product_id_emb_model',topN=topN)
# print(product_id_df)
# product_id_topn = f'product_id_top{topN}_l{product_id_size}'
# product_id_topn_path = f'{preprocess_path}/{product_id_topn}'
# product_id_df.to_pickle(product_id_topn_path)


# # %%
# label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')


# # %%
# train_df = train_df.merge(label_df,on='user_id')
# train_df = train_df.merge(industry_df,on='user_id')
# train_df = train_df.merge(advertiser_id_df,on='user_id')
# train_df = train_df.merge(product_id_df,on='user_id')
# print(train_df)
# print(train_df['age'].value_counts())
# print(train_df['gender'].value_counts())
# train_df.to_pickle("train5.pkl")


# # %%
# industry_test_df   = w2v(valid_df,'user_id','industry','train',16,'/data/workspace/kimi/tencent_ads/2020/kimi/industry_emb_model')
# print(industry_test_df)


# # %%
# advertiser_id_test_df   = w2v(valid_df,'user_id','advertiser_id','train',32,'/data/workspace/kimi/tencent_ads/2020/kimi/advertiser_id_emb_model')
# print(advertiser_id_test_df)


# # %%



# # %%
# label_test_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')


# # %%

# valid_df = valid_df.merge(label_df,on='user_id')
# valid_df = valid_df.merge(industry_test_df,on='user_id')
# valid_df = valid_df.merge(advertiser_id_test_df,on='user_id')
# print(valid_df)
# print(valid_df['age'].value_counts())
# print(valid_df['gender'].value_counts())
# valid_df.to_pickle("valid5.pkl")



# # %%



