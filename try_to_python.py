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


# %%
clk_df = pd.read_csv(f'{data_path}/train_preliminary/click_log.csv' )
print(clk_df)


# %%
user_click_sum_df = clk_df.groupby(['user_id']).click_times.sum().reset_index().rename(columns={'click_times':'click_times_total'})
print(user_click_sum_df)


# %%
user_day_count_df = clk_df.groupby(['user_id']).time.nunique().reset_index().rename(columns={'time':'active_days'})
print(user_day_count_df)


# %%
user_day_count_df = clk_df.groupby(['user_id']).time.nunique().reset_index().rename(columns={'time':'active_days'})
print(user_day_count_df)


# %%
user_df = user_day_count_df.merge(user_click_sum_df,on='user_id')
print(user_df)


# %%
ad_df = pd.read_csv(f'{data_path}/train_preliminary/ad.csv' )


# %%
merged_df = clk_df.merge(ad_df,on='creative_id')
#merged_df['wday'] = merged_df['time'].apply(lambda x :int(x /7))
#merged_df['month'] = merged_df['time'].apply(lambda x :int(x /30))
print(merged_df)
del clk_df


# %%

def tfidf(log,pivot,f,flag,L):
    #word2vec算法
    #log为曝光日志，以pivot为主键，f为embedding的对象，flag为dev或test，L是embedding的维度
    print("tdidf:",pivot,f)
    
    #构造文档
    log[f]=log[f].fillna(-1).astype(int)
    sentence=[]
    dic={}
    day=0
    log=log.sort_values(by='time')
    log['day']=log['time']
    for item in tqdm(log[['day',pivot,f]].values,total=len(log)):
#         if day!=item[0]:
#             for key in dic:
#                 sentence.append(dic[key])
#             dic={}
#             day=item[0]
        try:
            dic[item[1]].append(str(int(item[2])))
        except:
            dic[item[1]]=[str(int(item[2]))]
    for key in dic:
        sentence.append(" ".join(dic[key]))
    print(len(sentence))
    print(sentence[:3])
    #训练Word2Vec模型
    print('training...')
    random.shuffle(sentence)
    tfidf_list = TfidfVectorizer().fit_transform(sentence)
    print('outputing...')
    df_data = []
    for v1,v2 in zip(dic.keys(),tfidf_list):
        df_data.append([v1,v2.todense()])
    tfidf_df  = pd.DataFrame(df_data,columns=['user_id','tfidf'
                                              ])
    #保存文件
    return tfidf_df 

tfidf_df  = tfidf(merged_df,'user_id','advertiser_id','train',64)
print(tfidf_df)


# %%
print(tfidf_df)


# %%
train_user_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv' )
print(train_user_df)


# %%
user_final_df = user_df.merge(train_user_df,on='user_id')
user_final_df = user_final_df.merge(tfidf_df,on='user_idf')


