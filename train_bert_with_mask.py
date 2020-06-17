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

from gensim.models import callbacks


np.random.seed(2020)
random.seed(2020)

pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = 'dataset'
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

    def attention(self, query, key, value,mask,batch_size):
       score = tf.matmul(query, key, transpose_b=True)
       dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
       scaled_score = score / tf.math.sqrt(dim_key)
       reshaped_mask = tf.reshape(mask,(batch_size,1,1, -1))
       masked_score = scaled_score + reshaped_mask
       weights = tf.nn.softmax(masked_score, axis=-1)
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
       hidden ,mask  = inputs
       # x.shape = [batch_size, seq_len, embedding_dim]
       batch_size = tf.shape(hidden)[0]
       query = self.query_dense(hidden)  # (batch_size, seq_len, embed_dim)
       key = self.key_dense(hidden)  # (batch_size, seq_len, embed_dim)
       value = self.value_dense(hidden)  # (batch_size, seq_len, embed_dim)
       query = self.separate_heads(
           query, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       key = self.separate_heads(
           key, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       value = self.separate_heads(
           value, batch_size
       )  # (batch_size, num_heads, seq_len, projection_dim)
       attention, weights = self.attention(query, key, value,mask,batch_size)
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

    def call(self, inputs, training=False):
        hidden,mask = inputs
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(hidden + attn_output)
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
        print(f"Epoch {self.epoch}, loss: {epoch_loss} loss delta precent:{delta_precent} time: {time_taken//60}min {round(time_taken%60,3)}s")
        if  self.save and self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print(f"Better model. Best loss: {self.best_loss}")
            model.save(self.save_path)
            print(f"Model {self.save_path} save done!")
        self.pre_loss = cum_loss
        self.pre_epoch_loss = epoch_loss
        self.since = time.time()

def get_emb_mat(f,L=64,window=5,flag='clk_ns_total'):
    emb_dic_path = f'model/{f}_{flag}_s{L}_w{window}_emb_dict.json'
    emb_mat_path = f'model/{f}_{flag}_s{L}_w{window}_emb_mat.npy'
    emb_model_path = f'model/{f}_{flag}_s{L}_w{window}_emb.model'
    print(emb_model_path)
    if os.path.exists(emb_dic_path) and os.path.exists(emb_mat_path):
        with open(emb_dic_path,'r') as load_f:
            word_index = json.load(load_f)
        embedding_mat = np.load(emb_mat_path)
    else:
        emb_model = Word2Vec.load(emb_model_path)
        print(emb_model)
        te_df = pd.read_pickle( f'preprocess/{f}_target_encode.pkl')
        #te_df = te_df.set_index(f)
        print(te_df)
        te_dic = {}
        for row in tqdm(te_df.values,total=len(te_df)):
            te_dic[row[0]] = np.array(row[1:])
        vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语
        word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
        word_vector = {} # 初始化`[word : vector]`字典
        embedding_mat = np.zeros((len(vocab_list) + 1, emb_model.vector_size + 12 ))
        for i in tqdm(range(len(vocab_list)),total=len(vocab_list)):
            word = vocab_list[i]  # 每个词语
            word_index[word] = i + 1 # 词语：索引
            te_values = te_dic.get(word,np.array([0 for i in range(12) ]))
            word_vector[word] = np.concatenate((emb_model.wv[word], te_values), axis=0)  # 词语：词向量 + 目标编码
            embedding_mat[i + 1] = np.concatenate((emb_model.wv[word], te_values), axis=0)   # 词向量矩阵

        with open(emb_dic_path,'w') as fp:
            fp.write(json.dumps(word_index))

        np.save(emb_mat_path,embedding_mat)
        print(embedding_mat.shape)
    return word_index,embedding_mat

def get_seq_model(f,maxlen, vocab_size, emded_dim,embedding_mat=None,num_heads=2,ff_dim=32,output_dim=20,has_position=True):
    seq_inputs = layers.Input(shape=(maxlen,))
    mask_seq_inputs = layers.Input(shape=(maxlen,))
    embedding_layer =  TokenAndPositionEmbedding(f,maxlen,vocab_size,embed_dim + 12,embedding_mat=embedding_mat,has_position=has_position)
    seq_embedding= embedding_layer(seq_inputs)
    transformer_block =  TransformerBlock(embed_dim + 12, num_heads, ff_dim + 12 )
    x =  transformer_block([seq_embedding,mask_seq_inputs])
    #x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    return x, seq_inputs,mask_seq_inputs



def get_seq_data(seq_df,mask_seq_df,f,maxlen,L=192,flag='clk_times',dataset='train',window=8):
    word_index,embedding_mat = get_emb_mat(f,L=L,window=window,flag=flag)
    seq_values_path =  f'preprocess/{dataset}_{f}_{flag}_maxlen{maxlen}_int_seq.npy'
    seq_mask_path = f'preprocess/mask_{f}_seq.npy'
    print(seq_values_path)
    if os.path.exists(seq_values_path):
        seq_values = np.load(seq_values_path)
    else:
        result=[]
        hit=0
        miss=0
        print(seq_df)
        for row in tqdm(seq_df[['user_id',f'{f}_clk_times_seq']].values,total=len(seq_df)):
            try:
                result.append([row[0],[word_index[i]  for i in row[-1]]])
                hit+=1
            except Exception as e:
                miss+=1
        print(f'hit:{hit}, miss:{miss}')

        int_seq_df  = pd.DataFrame(result,columns=['user_id',f'{f}_int_seq'])
        seq_values = np.array(int_seq_df[[f'{f}_int_seq']].values[:,0])
        seq_values = keras.preprocessing.sequence.pad_sequences(seq_values, maxlen=maxlen)
        np.save(seq_values_path,seq_values)
    print(f"end pad features seq ")

    if os.path.exists(seq_mask_path):
        mask_seq_values = np.load(seq_mask_path)
    else:
        mask_seq_values = np.array(mask_seq_df[[f'{f}_clk_times_mask_seq']].values[:,0])
        mask_seq_values = keras.preprocessing.sequence.pad_sequences(mask_seq_values, maxlen=maxlen)
        np.save(seq_mask_path,mask_seq_values)
    print(f"end pad mask seq ")
    

    return  word_index,embedding_mat,seq_values,mask_seq_values


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

def get_label(flag='age'):
    label_df = pd.read_csv(f'{data_path}/user.csv')
    train_y = label_df[label_df.user_id <= 720000][[flag]].values
    valid_y = label_df[label_df.user_id > 720000][[flag]].values

    before_one_hot =  train_y.reshape([-1,1])
    enc = OneHotEncoder()
    enc.fit(before_one_hot)

    one_hoted_train_y  = enc.transform(before_one_hot).toarray()

    before_one_hot =  valid_y.reshape([-1,1])
    enc = OneHotEncoder()
    enc.fit(before_one_hot)

    one_hoted_valid_y  = enc.transform(before_one_hot).toarray()
    return one_hoted_train_y,one_hoted_valid_y
age_one_hoted_train_y, age_one_hoted_valid_y=  get_label()
gender_one_hoted_train_y, gender_one_hoted_valid_y=  get_label(flag='gender')

#####

maxlen = 150
embed_dim = 192  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = embed_dim  # Hidden layer size in feed forward network inside transformer
window = 15


train_x_list = []
valid_x_list = []
vocab_size_list = []
seq_output_list = []
seq_input_list = []

#for i in ['creative_id', 'advertiser_id']:
#for i in [ 'creative_id', 'ad_id','product_id','advertiser_id','industry']:
for i in [ 'creative_id']:
#for i in ['creative_id','product_id','advertiser_id','industry']:
    print(f'start {i}...')
    
    seq_df = pd.read_pickle(f'{preprocess_path}/{i}_clk_ns_total_clk_times_seq.pkl').sort_values(by='user_id')
    seq_df = seq_df[seq_df.user_id < 1000000]
    mask_seq_df = pd.read_pickle(f'{preprocess_path}/mask_{i}.pkl').sort_values(by='user_id')
    mask_seq_df = mask_seq_df[mask_seq_df.user_id < 1000000]
    word_index,embedding_mat,seq_values,mask_seq_values = get_seq_data(seq_df,mask_seq_df,i,maxlen,window=window,L=embed_dim)
    train_x_list.append(seq_values[:720000])
    valid_x_list.append(seq_values[720000:])
    train_x_list.append(mask_seq_values[:720000])
    valid_x_list.append(mask_seq_values[720000:])
    vocab_size_list.append(len(word_index))
    seq_embedding,seq_inputs,mask_seq_inputs = get_seq_model(i,maxlen,len(word_index),embed_dim,embedding_mat=embedding_mat,num_heads=num_heads,ff_dim=ff_dim)
    seq_output_list.append(seq_embedding)
    seq_input_list.append(seq_inputs)
    seq_input_list.append(mask_seq_inputs)
    print(f"end {i}...")
    gc.collect()

train_statics_df,valid_statics_df = get_statics()
statics_features_size = len(train_statics_df.columns)

train_x_list.append(train_statics_df)
valid_x_list.append(valid_statics_df)

def get_model(seq_input_list,seq_output_list, statics_features_size):
    statics_inputs = layers.Input(shape=(statics_features_size,))
    if len(seq_output_list) > 1:
        x = layers.concatenate( [i for i in seq_output_list])
    else:
        x = seq_output_list[0]

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)

    combined = layers.concatenate( [x,statics_inputs])
    age_outputs = layers.Dense(10, activation="softmax",name='age_outputs')(combined)
    gender_outputs = layers.Dense(2, activation="softmax",name='gender_outputs')(combined)


    model = keras.Model(inputs= seq_input_list + [statics_inputs], outputs=[age_outputs,gender_outputs])

    model.compile(loss={
                        'age_outputs':"categorical_crossentropy",
                        'gender_outputs':"categorical_crossentropy",
                        },
                    loss_weights= {
                             'age_outputs':0.6,
                        'gender_outputs':0.4,  
                    },
                 optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    return model

model =  get_model(seq_input_list,seq_output_list,statics_features_size)

mc =keras.callbacks.ModelCheckpoint(
    f'model/age_bert_lstm_weight_w{window}_checkpoint.h5',
    monitor="val_age_outputs_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch"
)

model.fit(train_x_list,
         [age_one_hoted_train_y,gender_one_hoted_train_y],
        validation_data=(valid_x_list,[age_one_hoted_valid_y, gender_one_hoted_valid_y]),
        #validation_steps=10,
        callbacks=[mc],
        shuffle=True,
        epochs=10)

model.save_weights('model/age_bert_weight_w{window}_checkpoint_end.h5')

ret = []
valid_predict=  model.predict(valid_x_list)

for i,age_percent in zip(range(len(valid_predict)),valid_predict):
    ret.append([i + 720001, age_percent])

ret_df = pd.DataFrame(ret,columns=['user_id','bert_percent'])
ret_df = ret_df.to_pickle(f'{preprocess_path}/bert_valid_ret.pkl')
print('end_udpate_valid')

ret = []
train_predict=  model.predict(train_x_list)

for i,age_percent in zip(range(len(train_predict)),train_predict):
    ret.append([i + 1, age_percent])

ret_df = pd.DataFrame(ret,columns=['user_id','bert_percent'])
print(ret_df)

ret_df = ret_df.to_pickle(f'{preprocess_path}/bert_train_ret.pkl')

