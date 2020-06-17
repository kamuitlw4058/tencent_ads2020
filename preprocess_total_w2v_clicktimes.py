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

# def get_merged_log(flag):
#     merged= f'{flag}_merged_log.pkl'
#     merged_path = f'{preprocess_path}/{merged}'
#     merged_df = pd.read_pickle(merged_path)
#     print(merged_df)
#     return merged_df

# train_merged_log_df = get_merged_log('train')
# test_merged_log_df = get_merged_log('test')
# total_merged_df = pd.concat([train_merged_log_df,test_merged_log_df]).sort_values(by='time')
# print(total_merged_df)

# del train_merged_log_df
# del test_merged_log_df
# gc.collect()
# total_merged_df.to_pickle(f'{preprocess_path}/total_merged_log.pkl')

# print("start to read merged log")
total_merged_df = pd.read_pickle(f'{preprocess_path}/total_merged_log.pkl')
# # %%


# %%
class EpochSaver(callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, path,save=True):
        self.save_path = path
        self.epoch = 0
        self.pre_loss = 0
        self.pre_epoch_loss = 0
        self.best_loss = 999999999.9
        self.total_since = time.time()
        self.since = time.time()
        self.save=save
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        if self.pre_epoch_loss != 0:
            delta  = epoch_loss -self.pre_epoch_loss
            delta_precent = delta / self.pre_epoch_loss
        else:
            delta_precent = 0
        print(f"Epoch {self.epoch},total loss:{cum_loss}  loss: {epoch_loss} loss delta precent:{delta_precent} time: {time_taken//60}min {round(time_taken%60,3)}s")
        if  self.save and self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print(f"Better model. Best loss: {self.best_loss}")
            model.save(self.save_path)
            print(f"Model {self.save_path} save done!")
        self.pre_loss = cum_loss
        self.pre_epoch_loss = epoch_loss
        self.since = time.time()

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
    model_word2vec.build_vocab(sentences)
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
            
        if i - last_opt_index > 10:
            print('model not opt break')
            break
            
    
    
    time_elapsed = time.time() - total_since
    print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_word2vec


# %%
def w2v(log,pivot,f,flag,L,model_path,seq_len=200,sentence_len=100,window=5,sg=1,negative=5,workers=10,iter=10):
    print("w2v:",pivot,f,model_path)

    #构造数据

    print('build data...')
    grouped_path =  f'{preprocess_path}/grouped_{f}.pkl'
    if os.path.exists(grouped_path):
        grouped_df = pd.read_pickle(grouped_path)
    else:
        if f != 'time':
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum','time':'max'}).reset_index().sort_values(by=['user_id','click_times','time'],ascending=[True, False,True])
        else:
            grouped_df =  log.groupby(['user_id', f]).agg({'click_times':'sum'}).reset_index().sort_values(by=['user_id','click_times'],ascending=[True, False])

        grouped_df.to_pickle(grouped_path)
    print(grouped_df)


    #构造Mask

    mask_path =  f'{preprocess_path}/mask_{f}.pkl'
    if not os.path.exists(mask_path):
        print('build mask...')
        dic={}
        for item in tqdm(grouped_df[[pivot,'click_times']].values,total=len(grouped_df)):
            try:
                dic[item[0]].append(int(item[1]))
            except:
                dic[item[0]]=[int(item[1])]

        ret = []
        for key in dic:
            ret.append([key,dic[key]])
        print(ret[:20])
        cols = ['user_id'] + [f'{f}_clk_times_mask_seq']
        ret_df = pd.DataFrame(ret,columns=cols)
        ret_df.to_pickle(mask_path)
        del ret_df
        del ret
        gc.collect()

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
    del grouped_df
    gc.collect()
    print(len(sentence))

    ##
    #训练Word2Vec模型
    #
    print('shuffle...')
    random.shuffle(sentence)
    print('training...')
    if isinstance(window,int):
        window = [window]
    for i in window:
        print(f'start training window:{i} workers:{workers} iter:{iter}')
        model_file_path = f'{model_path}/{f}_{flag}_s{L}_w{i}_emb.model'
        print(model_file_path)
        model = get_word2vec(sentence, model_file_path,L=L, window=i, workers=workers,sg=sg,negative=negative,iter=iter)
        #model.save(model_file_path)
        print(model)
        del model
        gc.collect()

        output_seq_path = f'{preprocess_path}/{f}_{flag}_clk_times_seq.pkl'
        if not os.path.exists(output_seq_path):
            ret = []
            for key in dic:
                ret.append([key,dic[key]])
            print(ret[:20])
            cols = ['user_id'] + [f'{f}_clk_times_seq']
            ret_df = pd.DataFrame(ret,columns=cols)
            ret_df.to_pickle(output_seq_path)





# %%
size=64
window = [15]
workers = 40
iter = 40
flag = 'clk_ns_total'
model_dir = f'model'

# %%
#for i in ['time', 'creative_id', 'ad_id','product_id','advertiser_id','product_category','industry']:
gc.collect()
for w in window:
    for i in [ 'creative_id']:
        w2v(total_merged_df,'user_id',i,flag,size,model_dir,window=w,iter=iter,workers=workers)
        gc.collect()


# %%



