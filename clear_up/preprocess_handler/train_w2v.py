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
base_path = '/newdata/worksapce/kimi/vs_code/tencent_ads/2020/kimi'
data_path = f'{base_path}/../dataset'
preprocess_path = f'{base_path}/preprocess'
model_dir_path = f'{base_path}/model'

df = pd.read_pickle(f'{preprocess_path}/filtered_merged_log.pkl')
print(df)
df = df.sort_values(by=['time'])
size = 64
window = 15



#feat_list = ['time','creative_id','ad_id','product_id','product_category','advertiser_id','industry']
feat_list = ['creative_id']
for i in feat_list:
    user_time_seq_dict_path = f'{preprocess_path}/user_time_seq_dict.json'
    if not os.path.exists(user_time_seq_dict_path):
        sentences_dic =  build_sentences_dict(df,'user_id',i)
        print(sentences_dic)
        user_time_seq_dict_json =  json.dumps(sentences_dic)
        with open(user_time_seq_dict_path,'w') as f:
            f.write(user_time_seq_dict_json)
        
    model_path = f'{model_dir_path}/{i}_s{size}_w{window}.model'
    model = word2vec(sentences_dic.values,model_path,window=15,iter=20)



