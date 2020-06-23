# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
from  collections import Counter

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
#base_path = '/newdata/worksapce/kimi/vs_code/tencent_ads/2020/kimi'
base_path = '/data/workspace/kimi/tencent_ads/2020/kimi'
data_path = f'{base_path}/../dataset'
preprocess_path = f'{base_path}/preprocess'



# %%
df = pd.read_pickle(f'{preprocess_path}/total_merged_log.pkl')
print(df)


# %%
grouped_user_id = df.groupby(['user_id'])['time'].count()
print(grouped_user_id)
filtered_grouped_user_id = grouped_user_id[grouped_user_id<= 150]
print(filtered_grouped_user_id)


# %%
filtered_df = df[df.user_id.isin(filtered_grouped_user_id.index)]
print(filtered_df)

filtered_df = filtered_df.sort_values(by=['time'])
filtered_df['user_id'] =  filtered_df['user_id'].astype(int).astype(str)
# %%
filtered_df.to_pickle(f'{preprocess_path}/filtered_merged_log.pkl')


# %%



