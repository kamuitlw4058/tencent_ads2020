
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
from sklearn.model_selection import KFold, StratifiedKFold


np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'



name = 'gender'
columns = ['']
train_log_df =  pd.read_pickle(f'{preprocess_path}/train_target_encoder_{name}.pkl').astype(float).fillna(0)
valid_log_df =  pd.read_pickle(f'{preprocess_path}/valid_target_encoder_{name}.pkl').astype(float).fillna(0)
features = []
for feat in ['product_id','product_category','advertiser_id','industry']:
    f =[f'{feat}_gender{i}_kfold_mean'  for i in range(2) ]
    features = features + f

agg_dict = dict(zip(features,[ ['min','max','mean','std'] for i in range(len(features)) ]))
print(agg_dict)



train_df = train_log_df.groupby('user_id').agg(agg_dict)
valid_df = valid_log_df.groupby('user_id').agg(agg_dict)
print(train_df)
print(valid_df)
train_df.to_pickle(f'{preprocess_path}/train_user_target_encoder_{name}.pkl')
valid_df.to_pickle(f'{preprocess_path}/valid_user_target_encoder_{name}.pkl')



