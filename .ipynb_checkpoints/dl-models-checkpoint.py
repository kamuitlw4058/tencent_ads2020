import numpy as np
import pandas as pd
#from keras.layers import *
#from keras.activations import softmax
#from keras.models import Model
#from keras.optimizers import Nadam, Adam
#from keras.regularizers import l2
#import keras.backend as K

import sys
import os
import pandas as pd
import numpy as np
import random
import gc
import math
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
#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
#from tensorflow.keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import  OneHotEncoder
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

MAX_LEN = 150

pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)

def create_pretrained_embedding(pretrained_weights, trainable=False, **kwargs):
    print(pretrained_weights)
    "Create embedding layer from a pretrained weights array"
    #pretrained_weights = np.load(pretrained_weights)
    in_dim, out_dim = pretrained_weights.shape
    print(f'create_pretrained_embedding :{in_dim}   out_dim:{out_dim}')
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding1, 
                           pretrained_embedding2, 
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933
    
    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))
    
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, 
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)]) 
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                  metrics=['binary_crossentropy','accuracy'])
    return model


def esim(pretrained_embedding1, 
         pretrained_embedding2, 
         maxlen=MAX_LEN, 
         lstm_dim=64, 
         dense_dim=64, 
         dense_dropout=0.5):
             
    # Based on arXiv:1609.06038
    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))
    
    # Embedding
    embedding1 = create_pretrained_embedding(pretrained_embedding1, mask_zero=False)
    embedding2 = create_pretrained_embedding(pretrained_embedding2, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding1(q1))
    q2_embed = bn(embedding2(q2))

    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    
    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)]) 
       
    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(10, activation="softmax")(dense)
    
    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def get_seq_data(f,maxlen,L=64):
    seq_df = pd.read_pickle(f'{preprocess_path}/{f}_s64_total_seq.pkl').sort_values(by='user_id')
    seq_df = seq_df[seq_df.user_id < 1000000]

    emb_model = Word2Vec.load(f'model/{f}_clk_emb.model_{L}')
    print(emb_model)

    vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语
    word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
    word_vector = {} # 初始化`[word : vector]`字典

    # 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
    # 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
    embedding_mat = np.zeros((len(vocab_list) + 1, emb_model.vector_size))

    for i in range(len(vocab_list)):
        # print(i)
        word = vocab_list[i]  # 每个词语
        word_index[word] = i + 1 # 词语：索引
        word_vector[word] = emb_model.wv[word] # 词语：词向量
        embedding_mat[i + 1] = emb_model.wv[word]  # 词向量矩阵

    print(embedding_mat.shape)

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
    print(int_seq_df)

    train_df  = int_seq_df[int_seq_df.user_id <=720000]
    valid_df = int_seq_df[int_seq_df.user_id > 720000]
    print(train_df)

    train_values = np.array(train_df[[f'{f}_int_seq']].values[:,0])
    valid_values = np.array(valid_df[[f'{f}_int_seq']].values[:,0])


    train_values = keras.preprocessing.sequence.pad_sequences(train_values, maxlen=maxlen)
    valid_values = keras.preprocessing.sequence.pad_sequences(valid_values, maxlen=maxlen)
    print(f"end pad seq")
    return train_values, valid_values , embedding_mat,len(vocab_list) + 1


train_x_list = []
valid_x_list = []

train_x_creative_id, valid_x_creative_id , embedding_matrix_creative_id,vocab_size_creative_id =  get_seq_data('creative_id',MAX_LEN)
train_x_advertiser_id, valid_x_advertiser_id , embedding_matrix_advertiser_id,vocab_size_advertiser_id =  get_seq_data('advertiser_id',MAX_LEN)

train_x_list = [train_x_creative_id,train_x_advertiser_id]
valid_x_list = [valid_x_advertiser_id,valid_x_advertiser_id]

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

model = esim(embedding_matrix_creative_id,embedding_matrix_advertiser_id)
# for i in ['time', 'creative_id', 'ad_id','product_id','advertiser_id','product_category','industry']:
#     print(f'start {i}...')
#     train_x, valid_x , embedding_matrix,vocab_size =  get_seq_data(i,maxlen)
#     train_x_list.append(train_x)
#     valid_x_list.append(valid_x)
#     vocab_size_list.append(vocab_size)
#     print(vocab_size)
#     seq_embedding,seq_inputs = get_seq_model(i,maxlen,vocab_size,embed_dim,embedding_mat=embedding_matrix,num_heads=num_heads,ff_dim=ff_dim)
#     seq_output_list.append(seq_embedding)
#     seq_input_list.append(seq_inputs)
#     print(f"end {i}...")
#     gc.collect()
model.summary()
mc =keras.callbacks.ModelCheckpoint(
    f'model/esim_age_weight_checkpoint.h5',
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
        #callbacks=[mc],
        shuffle=True,
        epochs=10)
