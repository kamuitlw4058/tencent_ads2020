# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

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

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


clk_train_file= 'click_log.pkl'
clk_train_file_path = f'{preprocess_path}/{clk_train_file}'
clk_train_file_input_path = f'train_preliminary/click_log.csv'



if not os.path.exists(clk_train_file_path):
    clk_log_df = pd.read_csv(f'{data_path}/{clk_train_file_input_path}' )
    clk_log_df.to_pickle(clk_train_file_path)
else: 
    clk_log_df = pd.read_pickle(clk_train_file_path)
    
#print(clk_log_df)



# clk_test_file= 'click_log_test.pkl'
# clk_test_file_path = f'{preprocess_path}/{clk_test_file}'
# if not os.path.exists(clk_test_file_path):
#     clk_test_df = pd.read_csv(f'{data_path}/test/click_log.csv' )
#     clk_test_df.to_pickle(clk_test_file_path)
# else: 
#     clk_test_df = pd.read_pickle(clk_test_file_path)
# print(clk_test_df)



def clk_active_days(clk_df,pivot,f,out_col,base_col='time',base_func=''):

    if base_func == 'nunique':
        active_days_df = clk_df.groupby([pivot,f])[base_col].nunique().reset_index().rename(columns={base_col:out_col})
    else:
        active_days_df = clk_df.groupby([pivot,f])[base_col].count().reset_index().rename(columns={base_col:out_col})

    active_days_max_df = active_days_df.groupby([pivot])[out_col].max().reset_index().rename(columns={out_col:f'{out_col}_max'})
    active_days_min_df = active_days_df.groupby([pivot])[out_col].min().reset_index().rename(columns={out_col:f'{out_col}_min'})
    active_days_mean_df = active_days_df.groupby([pivot])[out_col].mean().reset_index().rename(columns={out_col:f'{out_col}_mean'})
    active_days_std_df = active_days_df.groupby([pivot])[out_col].std().reset_index().rename(columns={out_col:f'{out_col}_std'})

    active_days_statistics_df = active_days_max_df.merge(active_days_min_df,on='user_id')
    active_days_statistics_df = active_days_statistics_df.merge(active_days_mean_df,on='user_id')
    active_days_statistics_df = active_days_statistics_df.merge(active_days_std_df,on='user_id')
    return active_days_statistics_df


def clk_statics(clk_df):
    clk_df['week'] = clk_df['time'].apply(lambda x: int(x/7))
    clk_df['month'] = clk_df['time'].apply(lambda x: int(x/30)) 
    #print(clk_df)

    user_click_sum_df = clk_df.groupby(['user_id']).click_times.sum().reset_index().rename(columns={'click_times':'click_times_total'})
    #print(user_click_sum_df)


    user_day_count_df = clk_df.groupby(['user_id']).time.nunique().reset_index().rename(columns={'time':'active_days'})
    #print(user_day_count_df)


    # user_log_day_clicks_df = clk_df.groupby(['user_id','time']).creative_id.count().reset_index().rename(columns={'creative_id':'day_clicks'})
    # #print(user_log_day_clicks_df)
    # user_day_clicks_max_df = user_log_day_clicks_df.groupby(['user_id']).day_clicks.max().reset_index().rename(columns={'day_clicks':'day_clicks_max'})
    # user_day_clicks_min_df = user_log_day_clicks_df.groupby(['user_id']).day_clicks.min().reset_index().rename(columns={'day_clicks':'day_clicks_min'})
    # user_day_clicks_mean_df = user_log_day_clicks_df.groupby(['user_id']).day_clicks.mean().reset_index().rename(columns={'day_clicks':'day_clicks_mean'})
    # user_day_clicks_std_df = user_log_day_clicks_df.groupby(['user_id']).day_clicks.std().reset_index().rename(columns={'day_clicks':'day_clicks_std'})


    # user_df = user_day_count_df.merge(user_click_sum_df,on='user_id')
    # user_df = user_df.merge(user_day_clicks_max_df,on='user_id')
    # user_df = user_df.merge(user_day_clicks_min_df,on='user_id')
    # user_df = user_df.merge(user_day_clicks_mean_df,on='user_id')
    # user_df = user_df.merge(user_day_clicks_std_df,on='user_id')
    active_day_df =  clk_active_days(clk_df,'user_id','time', 'active_days',base_col ='creative_id',base_func='nunique')
    #print(f"active_day_df\n{active_day_df}")

    week_active_day_df =  clk_active_days(clk_df,'user_id','week','week_active_days')
    #print(f"week_active_day_df:\n{week_active_day_df}")

    month_acitve_day_df =  clk_active_days(clk_df,'user_id','month','month_acitve_days')
    #print(f"month_active_day_df:\n{month_acitve_day_df}")

    user_df = user_click_sum_df.merge(user_day_count_df,on='user_id')
    user_df = user_df.merge(active_day_df,on='user_id')
    user_df = user_df.merge(week_active_day_df,on='user_id')
    user_df = user_df.merge(month_acitve_day_df,on='user_id')
    return user_df






user_df =  clk_statics(clk_log_df)
print(user_df)

user_df.to_pickle(f'{preprocess_path}/user_statics_train.pkl')