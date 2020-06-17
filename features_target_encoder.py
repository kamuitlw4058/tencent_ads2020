
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

name = 'creative_id'
label_name ='gender'
label_class = 2
t1_df =  pd.read_pickle(f'{preprocess_path}/train_target_encoder_v1.pkl' )
v1_df =  pd.read_pickle(f'{preprocess_path}/valid_target_encoder_v1.pkl' )

t1_df = t1_df[['user_id',name,'fold'] + [f'{label_name}{i}' for i in range(label_class)]].fillna(0)
v1_df = v1_df[['user_id',name]]

print(t1_df)
print(v1_df)


def get_kflod_targe_encoder(train_df_,valid_df_):
    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    kfold_features1 = []
    t_df = train_df_
    v_df = valid_df_

    for feat in [name]:

            nums_columns = [f'{label_name}{i}' for i in range(label_class)]
            for f in nums_columns:
                colname1 = feat + '_' + f + '_kfold_mean'
                print(feat,f,' mean')
                kfold_features1.append(colname1)

                t_df[colname1] = None
                for fold_,(trn_idx,val_idx) in enumerate(folds.split(t_df,t_df)):
                    Log_trn     = t_df.iloc[trn_idx]
                    # mean
                    order_label = Log_trn.groupby([feat])[f].mean()
                    tmp         = t_df.loc[t_df.fold==fold_,[feat]]
                    t_df.loc[t_df.fold==fold_, colname1] = tmp[feat].map(order_label)
                    del Log_trn
                    del order_label
                    del tmp


                v_df[colname1] = None
                order_label   = t_df.groupby([feat])[f].mean()
                v_df[colname1] = v_df[feat].map(order_label)
                del order_label
                gc.collect()

                print(t_df)
    return t_df,v_df,kfold_features1


train_df,valid_df,kfold_features  = get_kflod_targe_encoder(t1_df,v1_df)
print(train_df)
print(valid_df)

train_df =train_df[['user_id'] +kfold_features].fillna(0)
train_df.to_pickle(f'{preprocess_path}/train_target_encoder_{name}.pkl')

valid_df =valid_df[['user_id'] +kfold_features].fillna(0)
valid_df.to_pickle(f'{preprocess_path}/valid_target_encoder_{name}.pkl')

#features =[f'{name}_age{i}_kfold_mean'  for i in range(10) ]
#agg_dict = dict(zip(features,[ ['min','max','mean','std'] for i in range(10) ]))
#print(agg_dict)

#train_df = train_df.groupby('user_id').agg(agg_dict)
#valid_df = train_df.groupby('user_id').agg(agg_dict)
#print(train_df)
#print(valid_df)
#train_df.to_pickle(f'{preprocess_path}/train_user_target_encoder_{name}_{label_name}.pkl')
#valid_df.to_pickle(f'{preprocess_path}/valid_user_target_encoder_{name}_{label_name}.pkl')






