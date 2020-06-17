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
from racing.nlp import word2vec
from racing.nlp import build_sentences_dict

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'

def get_merged_log(flag):
    merged= f'{flag}_merged_log.pkl'
    merged_path = f'{preprocess_path}/{merged}'
    merged_df = pd.read_pickle(merged_path)
    print(merged_df)
    return merged_df

train_merged_log_df = get_merged_log('train')
label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')
# train_merged_log_df = train_merged_log_df.merge(label_df,on='user_id',how='left')

test_merged_log_df = get_merged_log('test')
#total_merged_df = pd.concat([train_merged_log_df,test_merged_log_df]).sort_values(by='time')
#print(total_merged_df)

# del train_merged_log_df
# del test_merged_log_df
# gc.collect()
# total_merged_df.to_pickle(f'{preprocess_path}/total_merged_log.pkl')

# print("start to read merged log")
#total_merged_df = pd.read_pickle(f'{preprocess_path}/total_merged_log.pkl')
# # %%

label_dic = {}

for row in tqdm(label_df.values,total=len(label_df)):
    label_dic[row[0]]=[f'age{row[1]}',f'gender{row[2]}']



def get_sentences_with_label(log,label_dic,pivot,f,L,window=5,age_label='age',gender_label='gender',flag='train'):
    print('build data...')
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
    
    ###
    # 构建文档
    #
    print('build docs...')
    sentence=[]
    dic={}
    for item in tqdm(grouped_df[[pivot,f]].values,total=len(grouped_df)):
                
        item_label = label_dic[item[0]]
        try:
            dic[item[0]].append(str(int(item[1])))
            if (len(dic[item[0]]) % int(window/2)) == 1:
                dic[item[0]].append(item_label[0])
                dic[item[0]].append(item_label[1])
        except:
            dic[item[0]]=[str(int(item[1])),item_label[0],item_label[1]]

    for key in dic:
        sentence.append(dic[key])
    print(sentence[:5])
    gc.collect()
    print(len(sentence))

    print('shuffle...')
    random.shuffle(sentence)
    return sentence

def get_sentences(log,pivot,f,L,window=5,flag='test'):
    print('build data...')

    grouped_path =  f'{preprocess_path}/grouped_{flag}_{f}.pkl'
    if  os.path.exists(grouped_path):
        grouped_df = pd.read_pickle(grouped_path)
    else:
        if f != 'time':
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum','time':'max'}).reset_index().sort_values(by=['user_id','click_times','time'],ascending=[True, False,True])
        else:
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum'}).reset_index().sort_values(by=['user_id','click_times'],ascending=[True, False])

        grouped_df.to_pickle(grouped_path)
    print(grouped_df)
    
    ###
    # 构建文档
    #
    print('build docs...')
    sentence=[]
    dic={}
    for item in tqdm(grouped_df[[pivot,f]].values,total=len(grouped_df)):
        try:
            dic[item[0]].append(str(int(item[1])))
        except:
            dic[item[0]]=[str(int(item[1]))]

    for key in dic:
        sentence.append(dic[key])
    print(sentence[:5])
    gc.collect()
    print(len(sentence))

    print('shuffle...')
    random.shuffle(sentence)
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


# %%
size=128
window = [15]
workers = 40
iter = 50
flag = 'clk_times'
model_dir = f'model'


gc.collect()
for w in window:
    for i in [ 'creative_id']:
        train_sentences =  get_sentences(train_merged_log_df,'user_id',i,size,window=w,flag='train')
        test_sentences =  get_sentences(test_merged_log_df,'user_id',i,size,window=w,flag='test')
        sentences = np.concatenate([train_sentences,test_sentences],axis=0)
        print(len(sentences))
        w2v(sentences,'user_id',i,flag,size,model_dir,window=w,iter=iter,workers=workers)
        gc.collect()
