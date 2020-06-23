import os
import pandas as pd
import numpy as np
import random
import gc
import json


from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.feature_extraction.text import TfidfVectorizer
from  collections import Counter
from racing.nlp import  build_sentences_dict
from racing.nlp import  word2vec

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 4)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
base_path = '/data/workspace/kimi/tencent_ads/2020/kimi'
data_path = f'{base_path}/../dataset'
preprocess_path = f'{base_path}/preprocess'
model_dir_path = f'{base_path}/model'

print("start read logs...")
filtered_merged_logs_pkl = f'{preprocess_path}/filtered_merged_log.pkl'
df = pd.read_pickle(filtered_merged_logs_pkl)
print(df)

size = 64
window = 15
worker = 30
flag='time'
pivot = 'user_id'

#feat_list = ['time','creative_id','ad_id','product_id','product_category','advertiser_id','industry']
feat_list = ['creative_id']
for i in feat_list:
    print(f'start {i}...')
    seq_name = f'filtered_{i}_{flag}_seq'
    seq_path = f'{preprocess_path}/{seq_name}.pkl'

    if not os.path.exists(seq_path):
        sentences_dic =  build_sentences_dict(df,pivot,i)
        seq_df_data = []
        for k,v in sentences_dic.items():
            seq_df_data.append([k,v])
        seq_df = pd.DataFrame(seq_df_data,names=[pivot,seq_name])
        seq_df.to_pickle(seq_path)
    else:
        seq_df =  pd.read_pickle(seq_path)
    print(seq_df)

    model_path = f'{model_dir_path}/{i}_s{size}_w{window}.model'
    print(model_path)


