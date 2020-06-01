
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

label_name ='gender'
label_class =2

for name in ['product_id','product_category','advertiser_id','industry']:
    flag ='test'
    features =[f'{name}_{label_name}{i}_kfold_mean'  for i in range(label_class) ]
    agg_dict = dict(zip(features,[ ['min','max','mean','std'] for i in range(label_class) ]))
    print(agg_dict)

    log_df =  pd.read_pickle(f'{preprocess_path}/{flag}_target_encoder_{name}_{label_name}.pkl').astype(float).fillna(0)



    user_df = log_df.groupby('user_id').agg(agg_dict)
    print(user_df)
    user_df.to_pickle(f'{preprocess_path}/{flag}_user_target_encoder_{name}_{label_name}.pkl')
    gc.collect()



