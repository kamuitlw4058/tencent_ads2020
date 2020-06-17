# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import sys
import os
import pandas as pd
import numpy as np
import random
import gc
import math
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  collections import Counter
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import  OneHotEncoder

print(sys.getrecursionlimit())

sys.setrecursionlimit(500000)
np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
       super(MultiHeadSelfAttention, self).__init__()
       self.embed_dim = embed_dim
       self.num_heads = num_heads
       if embed_dim % num_heads != 0:
           raise ValueError(
               f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
           )
       self.projection_dim = embed_dim // num_heads
       self.query_dense = layers.Dense(embed_dim)
       self.key_dense = layers.Dense(embed_dim)
       self.value_dense = layers.Dense(embed_dim)
       self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
       score = tf.matmul(query, key, transpose_b=True)
       dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
       scaled_score = score / tf.math.sqrt(dim_key)
       weights = tf.nn.softmax(scaled_score, axis=-1)
       output = tf.matmul(weights, value)
       return output, weights

    def separate_heads(self, x, batch_size):
       x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
       return tf.transpose(x, perm=[0, 2, 1, 3])

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    def call(self, inputs):
       # x.shape = [batch_size, seq_len, embedding_dim]
       batch_size = tf.shape(inputs)[0]
       query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
       key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
       value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
       query = self.separate_heads(
           query, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       key = self.separate_heads(
           key, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       value = self.separate_heads(
           value, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       attention, weights = self.attention(query, key, value)
       attention = tf.transpose(
           attention, perm=[0, 2, 1, 3]
       )  # (batch_size, seq_len, num_heads, projection_dim)
       concat_attention = tf.reshape(
           attention, (batch_size, -1, self.embed_dim)
       )  # (batch_size, seq_len, embed_dim)
       output = self.combine_heads(
           concat_attention
       )  # (batch_size, seq_len, embed_dim)
       return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self,f, maxlen, vocab_size, embed_dim,embedding_mat=None,has_position=True):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.f = f
        self.has_position = has_position
        embeding_path =  f'embedding_model/{f}.npy'
        if embedding_mat is None:
            embedding_mat = np.load(embeding_path)
        else:
            np.save(embeding_path,embedding_mat)

        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                            embeddings_initializer=keras.initializers.Constant(embedding_mat),
                                            output_dim=embed_dim,
                                            trainable=False
                                            )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):

        emb = self.token_emb(x)
        if self.has_position:
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            return emb + positions
        else:
            return emb

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'f': self.f,
            'has_position':self.has_position
        })
        return config






def get_emb_mat(f,L=128,window=10,flag='clk_ns_total'):
    emb_dic_path = f'model/{f}_{flag}_s{L}_w{window}_emb_dict.json'
    emb_mat_path = f'model/{f}_{flag}_s{L}_w{window}_emb_mat.npy'
    if os.path.exists(emb_dic_path) and os.path.exists(emb_mat_path):
        with open(emb_dic_path,'r') as load_f:
            word_index = json.load(load_f)
        embedding_mat = np.load(emb_mat_path)
    else:
        emb_model = Word2Vec.load(f'model/{f}_{flag}_s{L}_w{window}_emb.model')
        print(emb_model)

        vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语
        word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
        word_vector = {} # 初始化`[word : vector]`字典

        embedding_mat = np.zeros((len(vocab_list) + 1, emb_model.vector_size))

        for i in range(len(vocab_list)):
            # print(i)
            word = vocab_list[i]  # 每个词语
            word_index[word] = i + 1 # 词语：索引
            word_vector[word] = emb_model.wv[word] # 词语：词向量
            embedding_mat[i + 1] = emb_model.wv[word]  # 词向量矩阵

        with open(emb_dic_path,'w') as fp:
            fp.write(json.dumps(word_index))

        np.save(emb_mat_path,embedding_mat)
        print(embedding_mat.shape)
    return word_index,embedding_mat


def get_seq_model(f,maxlen, vocab_size, emded_dim,embedding_mat=None,num_heads=2,ff_dim=32,output_dim=20,has_position=True):
    seq_inputs = layers.Input(shape=(maxlen,))
    embedding_layer =  TokenAndPositionEmbedding(f,maxlen,vocab_size,embed_dim,embedding_mat=embedding_mat,has_position=has_position)
    seq_embedding= embedding_layer(seq_inputs)
    return seq_embedding, seq_inputs


def get_seq_data(seq_df,f,maxlen,L=128,flag='clk_ns_total',dataset='train'):
    word_index,embedding_mat = get_emb_mat(f,L=L,flag=flag)
    seq_values_path =  f'preprocess/{dataset}_{f}_{flag}_maxlen{maxlen}_int_seq.npy'
    if os.path.exists(seq_values_path):
        seq_values = np.load(seq_values_path)
    else:
        result=[]
        hit=0
        miss=0
        for row in tqdm(seq_df[['user_id',f'{f}_seq']].values,total=len(seq_df)):
            try:
                result.append([row[0],[word_index[i]  for i in row[-1]]])
                hit+=1
            except Exception as e:
                miss+=1
        print(f'hit:{hit}, miss:{miss}')

        int_seq_df  = pd.DataFrame(result,columns=['user_id',f'{f}_int_seq'])

        seq_values = np.array(int_seq_df[[f'{f}_int_seq']].values[:,0])
        seq_values = keras.preprocessing.sequence.pad_sequences(seq_values, maxlen=maxlen)
        print(f"end pad seq ")
        np.save(seq_values_path,seq_values)

    return  word_index,embedding_mat,seq_values




def get_statics():
    train_statics_path = f'{preprocess_path}/train_user_base_statics_total.pkl'
    valid_statics_path = f'{preprocess_path}/valid_user_base_statics_total.pkl'
    if os.path.exists(train_statics_path) and os.path.exists(valid_statics_path):
        train_statics_df = pd.read_pickle(train_statics_path)
        valid_statics_df = pd.read_pickle(valid_statics_path)
    else:
        user_base_statics_df= pd.read_pickle(f'{preprocess_path}/train_user_base_statics.pkl')
        user_base_statics_df.columns = ['_'.join(i) for i in user_base_statics_df.columns.values]
        user_base_statics_df['click_times_sum_log'] = user_base_statics_df['click_times_sum'].apply(lambda x :math.log(x))
        user_base_statics_df['click_times_count_log'] = user_base_statics_df['click_times_count'].apply(lambda x :math.log(x))
        user_base_statics_df = user_base_statics_df.drop(['click_times_sum','click_times_count'],axis=1)
        user_base_statics_df = user_base_statics_df.astype(float).reset_index()
        print(user_base_statics_df)

        train_statics_df =  user_base_statics_df[user_base_statics_df.user_id <= 720000]
        valid_statics_df =  user_base_statics_df[user_base_statics_df.user_id > 720000]
        print(valid_statics_df)

        def merge_features(train_df,valid_df,train_file,valid_file,target_encode=False):
            train_features_df  = pd.read_pickle(f'{preprocess_path}/{train_file}')
            valid_features_df = pd.read_pickle(f'{preprocess_path}/{valid_file}')
            if target_encode:
                train_features_df.columns = [ '_'.join(i) for i in train_features_df.columns.values  ]
                valid_features_df.columns = ['_'.join(i) for i in valid_features_df.columns.values  ]

            train_df = train_df.merge(train_features_df,on='user_id')
            valid_df = valid_df.merge(valid_features_df,on='user_id')
            return train_df,valid_df

        for i in ['creative_id','ad_id', 'product_id','advertiser_id','industry','product_category']:
            print(f'merge {i}...')
            train_statics_df,valid_statics_df = merge_features(train_statics_df,valid_statics_df,f'train_user_target_encoder_{i}_age.pkl',f'valid_user_target_encoder_{i}.pkl',True)
            print(train_statics_df)
            print(valid_statics_df)

        train_statics_df = train_statics_df.drop(['user_id'],axis=1)
        valid_statics_df = valid_statics_df.drop(['user_id'],axis=1)

        train_statics_df.to_pickle(train_statics_path)
        valid_statics_df.to_pickle(valid_statics_path)
    return train_statics_df,valid_statics_df

def get_label():
    label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')
    train_y = label_df[label_df.user_id <= 720000][['age']].values
    valid_y = label_df[label_df.user_id > 720000][['age']].values

    before_one_hot =  train_y.reshape([-1,1])
    enc = OneHotEncoder()
    enc.fit(before_one_hot)

    one_hoted_train_y  = enc.transform(before_one_hot).toarray()

    before_one_hot =  valid_y.reshape([-1,1])
    enc = OneHotEncoder()
    enc.fit(before_one_hot)

    one_hoted_valid_y  = enc.transform(before_one_hot).toarray()
    return one_hoted_train_y,one_hoted_valid_y
one_hoted_train_y, one_hoted_valid_y=  get_label()

#####

maxlen = 150
embed_dim = 128  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer


train_x_list = []
valid_x_list = []
vocab_size_list = []
seq_output_list = []
seq_input_list = []

#for i in ['creative_id', 'advertiser_id']:
#for i in ['time', 'creative_id', 'ad_id','product_id','advertiser_id','product_category','industry']:
for i in ['creative_id','product_id','advertiser_id','industry']:
    print(f'start {i}...')
    seq_df = pd.read_pickle(f'{preprocess_path}/{i}_s64_total_seq.pkl').sort_values(by='user_id')
    seq_df = seq_df[seq_df.user_id < 1000000]
    word_index,embedding_mat,seq_values = get_seq_data(seq_df,i,maxlen)
    train_x_list.append(seq_values[:720000])
    valid_x_list.append(seq_values[720000:])
    vocab_size_list.append(len(word_index))
    seq_embedding,seq_inputs = get_seq_model(i,maxlen,len(word_index),embed_dim,embedding_mat=embedding_mat,num_heads=num_heads,ff_dim=ff_dim)
    seq_output_list.append(seq_embedding)
    seq_input_list.append(seq_inputs)
    print(f"end {i}...")
    gc.collect()


train_statics_df,valid_statics_df = get_statics()
statics_features_size = len(train_statics_df.columns)

train_x_list.append(train_statics_df)
valid_x_list.append(valid_statics_df)

def get_model(seq_feat_number, statics_features_size):

    statics_inputs = layers.Input(shape=(statics_features_size,))

    transformer_block =  TransformerBlock(embed_dim * seq_feat_number , num_heads, ff_dim * seq_feat_number )
    transformer_block2 =  TransformerBlock(embed_dim * seq_feat_number , num_heads, ff_dim *seq_feat_number )
    #transformer_block3 =  TransformerBlock(embed_dim * len(seq_output_list) , num_heads, ff_dim * len(seq_output_list) )
    if len(seq_output_list) > 0:
        x = layers.concatenate( [i for i in seq_output_list])
        x = transformer_block(x)
    else:
        x = transformer_block(seq_output_list[0])
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    #x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)

    combined = layers.concatenate( [x,statics_inputs])
    outputs = layers.Dense(10, activation="softmax")(combined)

    model = keras.Model(inputs= seq_input_list + [statics_inputs], outputs=outputs)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

model =  get_model(len(seq_output_list),statics_features_size)

mc =keras.callbacks.ModelCheckpoint(
    f'model/age_transform_lstm_weight_checkpoint.h5',
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch"
)

model.fit(train_x_list,
         one_hoted_train_y,
        validation_data=(valid_x_list,one_hoted_valid_y),
        #validation_steps=10,
        callbacks=[mc],
        shuffle=True,
        epochs=100)

model.save_weights('model/age_transform_lstm_weight_checkpoint_end.h5')


