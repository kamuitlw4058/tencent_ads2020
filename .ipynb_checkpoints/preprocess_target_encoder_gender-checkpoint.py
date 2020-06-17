
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

merged_df = pd.read_pickle(f'{preprocess_path}/train_merged_log.pkl' )
label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')
merged_df = merged_df.merge(label_df,on='user_id',how='left')
merged_df['gender'] = merged_df['gender'] -1
print(merged_df)



for i in range(2):
    merged_df[f'gender{i}']= None
    merged_df.loc[merged_df.gender == i,[f'gender{i}']] = 1
    merged_df[f'gender{i}'] = merged_df[f'gender{i}'].fillna(0)

print(merged_df)

train_df = merged_df[merged_df.user_id <= 720000]
valid_df = merged_df[merged_df.user_id > 720000]
t1_df = train_df.copy().reset_index()
v1_df = valid_df.copy().reset_index()
folds = KFold(n_splits=5, shuffle=True, random_state=2020)
t1_df['fold'] = None
for fold_,(trn_idx,val_idx) in enumerate(folds.split(t1_df,t1_df)):
    t1_df.loc[val_idx, 'fold'] = fold_
print(t1_df)
print(v1_df)
t1_df.to_pickle(f'{preprocess_path}/train_target_encoder_gender_v1.pkl')
v1_df.to_pickle(f'{preprocess_path}/valid_target_encoder_gender_v1.pkl')

