import pandas as pd


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
data_path = '/home/kimi/workspace/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


log_df = pd.read_csv(f'{data_path}/train_preliminary/click_log.csv')
ad_df = pd.read_csv(f'{data_path}/train_preliminary/ad.csv')

train_merged_df = log_df.merge(ad_df,on='user_id')
train_merged_df.to_pickle(f'{preprocess_path}/train_merged_log.pkl')
