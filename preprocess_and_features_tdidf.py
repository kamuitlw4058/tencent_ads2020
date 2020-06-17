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


np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'


merged_df = pd.read_pickle(f'{preprocess_path}/train_merged_log.pkl' )
label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')
merged_df = merged_df.merge(label_df,on='user_id',how='left')
merged_df['age'] = merged_df['age'] -1
merged_df['gender'] = merged_df['gender'] -1
print(merged_df)

def tfidf(log,pivot,f,flag,L):
    #word2vec算法
    #log为曝光日志，以pivot为主键，f为embedding的对象，flag为dev或test，L是embedding的维度
    print("tdidf:",pivot,f)
    
    #构造文档
    log[f]=log[f].fillna(-1).astype(str)
    sentence=[]
    dic={}
    day=0
    log=log.sort_values(by='time')
    log['day']=log['time']
    for item in tqdm(log[['day',pivot,f]].values,total=len(log)):
        try:
            dic[item[1]].append(str(item[2]))
        except:
            dic[item[1]]=[str(item[2])]
    for key in dic:
        sentence.append(" ".join(dic[key]))
    print(len(sentence))
    print(sentence[:3])
    #训练Word2Vec模型
    print('training...')
    #random.shuffle(sentence)
    tfidf_list = TfidfVectorizer(min_df=30,max_features=100000).fit_transform(sentence)
    print('outputing...')
    df_data = []
    arr_len = 0
    for v1,v2 in zip(list(dic.keys()),tfidf_list):
        arr = np.array(v2.todense()).flatten().tolist()
        if arr_len == 0:
            arr_len = len(arr)
            print(arr_len)
        df_data.append([v1] + arr)
    cols = ['user_id'] + [f'tfidf_{i}'  for i in range(arr_len)]
    tfidf_df= pd.DataFrame(df_data,columns=cols)
    #保存文件
    return tfidf_df 


# %%
tfidf_df  = tfidf(merged_df,'user_id','ad_id','train',64)
print(tfidf_df)
print(tfidf_df.shape)
tfidf_df.to_pickle("preprocess/train_tfidf_ad_id_age.pkl")


