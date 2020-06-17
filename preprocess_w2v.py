# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import pandas as pd
import numpy as np
import random
import gc
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import callbacks
from gensim.models.word2vec import LineSentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  collections import Counter
from racing.nlp.w2v import word2vec
from racing.nlp.w2v import build_sentences_dict

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = 'dataset'
preprocess_path = 'preprocess'

{
    'grouped_df':{
        'output_path':'123123',
        'input_path':'123',
        'opt':None
    }
}

def get_merged_log(flag):
    merged= f'{flag}_merged_log.pkl'
    merged_path = f'{preprocess_path}/{merged}'
    merged_df = pd.read_pickle(merged_path)
    print(merged_df)
    return merged_df

train_merged_log_df = get_merged_log('train')
test_merged_log_df = get_merged_log('test')
label_df = pd.read_csv(f'{data_path}/user.csv')

label_dic = {}

for row in tqdm(label_df.values,total=len(label_df)):
    label_dic[row[0]]=[f'age{row[1]}',f'gender{row[2]}']


def get_grouped(f,flag):
    grouped_path =  f'{preprocess_path}/grouped_{flag}_{f}.pkl'
    if os.path.exists(grouped_path):
        grouped_df = pd.read_pickle(grouped_path)
    else:
        if f != 'time':
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum','time':'max'}).reset_index().sort_values(by=['user_id','click_times','time'],ascending=[True, False,True])
        else:
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum'}).reset_index().sort_values(by=['user_id','click_times'],ascending=[True, False])

        grouped_df.to_pickle(grouped_path)
    print(grouped_df)
    return grouped_df
    

def get_sentences(log,pivot,f,L,window=5,flag='train',label_dic =None):
    print('build data...')
    grouped_df = get_grouped(f,flag)
    sentence = []
    dic = build_sentences_dict(grouped_df,pivot,f,interval=window/2,global_dic=label_dic)

    for key in dic:
        sentence.append(dic[key])
    print(sentence[:5])
    print(len(sentence))
    return sentence


def w2v(sentences,pivot,f,flag,L,model_path,seq_len=200,sentence_len=100,window=5,sg=1,negative=5,workers=10,iter=10):
    ##
    #训练Word2Vec模型
    #
    print('shuffle...')
    random.shuffle(sentences)
    print('training...')
    print(len(sentences))
    if isinstance(window,int):
        window = [window]
    for i in window:
        print(f'start training window:{i} workers:{workers} iter:{iter}')
        model_file_path = f'{model_path}/{f}_{flag}_s{L}_w{i}_emb.model'
        print(model_file_path)
        model = word2vec(sentences, model_file_path,L=L, window=i, workers=workers,sg=sg,negative=negative,iter=iter,cache=False)
        print(model)
        del model
        gc.collect()


size=192
window = [15]
workers = 40
iter = 50
flag = 'clk_times'
model_dir = f'model'
pivot = 'user_id'

gc.collect()
for w in window:
    for i in [ 'creative_id']:
        train_sentences =  get_sentences(train_merged_log_df,'user_id',i,size,window=w,flag='train')
        test_sentences =  get_sentences(test_merged_log_df,'user_id',i,size,window=w,flag='test')
        sentences = np.concatenate([train_sentences,test_sentences],axis=0)
        print(len(sentences))
        w2v(sentences,'user_id',i,flag,size,model_dir,window=w,iter=iter,workers=workers)
        gc.collect()



    