
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



label_name = 'gender'
lable_class = 2

def get_kflod_targe_encoder(train_df_,test_df_):
    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    kfold_features1 = []
    t_df = train_df_
    test_df = test_df_

    #for feat in ['product_id','product_category','advertiser_id','industry']:
    for feat in [agg_features]:

            nums_columns = [f'{label_name}{i}' for i in range(lable_class)]
            for f in nums_columns:
                colname1 = feat + '_' + f + '_kfold_mean'
                print(feat,f,' mean/median...')
                kfold_features1.append(colname1)

                test_df[colname1] = None
                order_label   = t_df.groupby([feat])[f].mean()
                test_df[colname1] = test_df[feat].map(order_label)
                del order_label
                gc.collect()

                print(test_df)
    return t_df,test_df,kfold_features1



t1_df =  pd.read_pickle(f'{preprocess_path}/train_target_encoder_gender_v1.pkl' )
test_df = pd.read_pickle(f'{preprocess_path}/test_merged_log.pkl' )

print(t1_df)
print(test_df)


#for i in ['ad_id']:
for i in ['creative_id','ad_id', 'product_id','advertiser_id','industry','product_category']:
#for i in ['ad_id', 'product_id','advertiser_id','industry']:
    print(f"start {i}...")
    agg_features = i
    #for feat in ['product_id','product_category','advertiser_id','industry']:
    featues = ['user_id','fold'] + [f'{label_name}{i}'  for i in range(lable_class) ] + [agg_features]

    t2_df = t1_df[featues]

    test_features = ['user_id'] + [agg_features]
    test1_df = test_df[test_features]

    train_df,test1_df,kfold_features  = get_kflod_targe_encoder(t2_df,test1_df)
    print(test_df)

    test1_df =test1_df[['user_id'] +kfold_features]
    test1_df = test1_df.fillna(0)
    test1_df.to_pickle(f'{preprocess_path}/test_target_encoder_{agg_features}_{label_name}.pkl')
    gc.collect()







