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
data_path = 'dataset'
preprocess_path = 'preprocess'

def get_merged_log(flag):
    merged= f'{flag}_merged_log.pkl'
    merged_path = f'{preprocess_path}/{merged}'
    merged_df = pd.read_pickle(merged_path)
    print(merged_df)
    return merged_df

train_merged_log_df = get_merged_log('train')
label_df = pd.read_csv(f'{data_path}/user.csv')
# train_merged_log_df = train_merged_log_df.merge(label_df,on='user_id',how='left')

test_merged_log_df = get_merged_log('test')
total_merged_df = pd.concat([train_merged_log_df,test_merged_log_df]).sort_values(by='time')
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





def deepwalk(log,f1,f2,flag,L,model_path,workers=40,sg=1,negative=5,iter=20,window=10):
    model_file_path = f'{model_path}/{f2}_{flag}_s{L}_w{window}_deepwalk_emb.model'
    #Deepwalk算法，
    print("deepwalk:",f1,f2)
    #构建图
    dic={}
    for item in log[[f1,f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))]=set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))]=set(['item_'+str(int(item[1]))])
    dic_cont={}
    for key in dic:
        dic[key]=list(dic[key])
        dic_cont[key]=len(dic[key])
    print("creating")
    #构建路径
    path_length=50
    sentences=[]
    length=[]
    for key in dic:
        sentence=[key]
        while len(sentence)!=path_length:
            key=dic[sentence[-1]][random.randint(0,dic_cont[sentence[-1]]-1)]
            if len(sentence)>=2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences)%100000==0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    #训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = get_word2vec(sentences, model_file_path,L=L, window=window, workers=workers,sg=sg,negative=negative,iter=iter)
    print('outputing...')
        #输出
    values=set(log[f1].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['user_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f1]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_dw_emb_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(f'{preprocess_path}/' +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')
    ########################
    values=set(log[f2].values)
    w2v=[]
    for v in values:
        try:
            a=[int(v)]
            a.extend(model['item_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df=pd.DataFrame(w2v)
    names=[f2]
    for i in range(L):
        names.append(f1+'_'+ f2+'_'+names[0]+'_dw_emb_'+str(L)+'_'+str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle(f'{preprocess_path}/' +f1+'_'+ f2+'_'+f2 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')


size=128
window = [15]
workers = 6
iter = 40
flag = 'total'
model_dir = f'model'

gc.collect()
for w in window:
    for i in [ 'creative_id']:
        print(f'start training window:{i} workers:{workers} iter:{iter}')
        deepwalk(total_merged_df,'user_id',i,flag,size,model_dir,window=w,iter=iter,workers=workers)
        gc.collect()

