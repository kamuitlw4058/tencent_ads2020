
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



label_name = 'age'
lable_class = 10

def get_kflod_targe_encoder(train_df_,agg_features):
    kfold_features1 = []
    t_df = train_df_

    for feat in [agg_features]:

            nums_columns = [f'{label_name}{i}' for i in range(lable_class)]
            for f in nums_columns:
                colname1 = feat + '_' + f + '_mean'
                print(feat,f,' mean...')
                order_label   = t_df.groupby([feat])[f].mean()
                order_label.to_pickle(f'{preprocess_path}/target_encoder_{colname1}_{label_name}.pkl' )


t1_df =  pd.read_pickle(f'{preprocess_path}/train_target_encoder_gender_v1.pkl' )

print(t1_df)


for i in ['creative_id','ad_id', 'product_id','advertiser_id','industry','product_category']:
    print(f"start {i}...")
    agg_features = i
    featues = ['user_id'] + [f'{label_name}{i}'  for i in range(lable_class) ] + [agg_features]

    t2_df = t1_df[featues]

    get_kflod_targe_encoder(t2_df,agg_features)








