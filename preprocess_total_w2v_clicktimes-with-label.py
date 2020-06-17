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

# label_dic = {}

# for row in tqdm(label_df.values,total=len(label_df)):
#     label_dic[row[0]]=[f'age{row[1]}',f'gender{row[2]}']




def get_word2vec(sentences,path, L=64,window=5,sg=1,negative=5,workers=10,iter=10):
    vector_size=L
    model_word2vec = Word2Vec(min_count=1,
                            window=window,
                            size=vector_size,
                            workers=workers,
                            sg=sg,
                            negative=negative,
                            iter=iter
                           )
    # 2, 遍历一遍语料库
    since = time.time()
    r = model_word2vec.build_vocab(sentences)
    print(r)
    time_elapsed = time.time() - since
    print('Time to build vocab: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # 3, 训练
    total_since = time.time()
    
    learning_rate = 0.5
    step_size = (0.5 - 0.001) / iter
    pre_epoch_loss = 0
    best_loss = 999999999.9
    last_opt_index = 0

    for i in range(iter):
        since = time.time()
        trained_word_count, raw_word_count = model_word2vec.train(sentences, compute_loss=True,
                                                         start_alpha=learning_rate,
                                                         end_alpha=learning_rate,
                                                         total_examples=model_word2vec.corpus_count,
                                                         epochs=1)
        epoch_loss = model_word2vec.get_latest_training_loss()
        time_taken = time.time() - since
        if pre_epoch_loss != 0:
            delta  = epoch_loss - pre_epoch_loss
            delta_precent = round(delta / pre_epoch_loss,5)
        else:
            delta_precent = 0
        print(f"Epoch {i+1}, loss: {epoch_loss} loss delta precent:{delta_precent} time: {time_taken//60}min {round(time_taken%60,3)}s")
        learning_rate -= step_size
        pre_epoch_loss = epoch_loss
        if  best_loss > epoch_loss:
            best_loss = epoch_loss
            last_opt_index = i
            print(f"Better model. Best loss: {best_loss}")
            model_word2vec.save(path)
            print(f"Model {path} save done!")
            
        if i - last_opt_index > 3:
            print('model not opt break')
            break
            
    
    
    time_elapsed = time.time() - total_since
    print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_word2vec

def get_sentences_with_label(log,label_dic,pivot,f,L,window=5,age_label='age',gender_label='gender'):
    print('build data...')
    grouped_path =  f'{preprocess_path}/grouped_train_{f}.pkl'
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


def get_sentences(log,pivot,f,L,window=5):
    print('build data...')

    grouped_path =  f'{preprocess_path}/grouped_test_{f}.pkl'
    if False and  os.path.exists(grouped_path):
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
        model = get_word2vec(sentences, model_file_path,L=L, window=i, workers=workers,sg=sg,negative=negative,iter=iter)
        print(model)
        del model
        gc.collect()



# %%
size=64
window = [10]
workers = 40
iter = 40
flag = 'clk_times'
model_dir = f'model'

gc.collect()
for w in window:
    for i in [ 'creative_id']:
        train_sentences =  get_sentences(train_merged_log_df,'user_id',i,size,window=w)
        test_sentences =  get_sentences(test_merged_log_df,'user_id',i,size,window=w)
        sentences = np.concatenate([train_sentences,test_sentences],axis=0)
        print(len(sentences))
        w2v(sentences,'user_id',i,flag,size,model_dir,window=w,iter=iter,workers=workers)
        gc.collect()



