# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import  OneHotEncoder

np.random.seed(2019)
random.seed(2019)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 280)
pd.set_option('display.max_colwidth', 150)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'
preprocess_path = 'preprocess'


# %%
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


# %%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
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


# %%
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim,embedding_matrix=None):
        super(TokenAndPositionEmbedding, self).__init__()
        if embedding_matrix is not None:
            self.token_emb = layers.Embedding(input_dim=vocab_size, 
                                            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                            output_dim=emded_dim,
                                            trainable=False
                                            )
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size, 
                                            output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


user_base_statics_df= pd.read_pickle(f'{preprocess_path}/train_user_base_statics.pkl')
user_base_statics_df.columns = ['_'.join(i) for i in user_base_statics_df.columns.values]
#label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')
user_base_statics_df['click_times_sum_log'] = user_base_statics_df['click_times_sum'].apply(lambda x :math.log(x))
user_base_statics_df['click_times_count_log'] = user_base_statics_df['click_times_count'].apply(lambda x :math.log(x))
user_base_statics_df = user_base_statics_df.drop(['click_times_sum','click_times_count'],axis=1)
user_base_statics_df = user_base_statics_df.astype(float).reset_index()
print(user_base_statics_df)




seq_df = pd.read_pickle(f'{preprocess_path}/ad_id_s64_total_seq.pkl')
print(seq_df)
seq_df = seq_df[seq_df.user_id < 1000000]


label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')


total_df = seq_df.merge(label_df,on='user_id',how='left')
print(total_df)
print(total_df.columns)



L = 64
emb_model = Word2Vec.load(f'model/ad_id_emb.model_{L}')
print(emb_model)
import numpy as np

vocab_list = [word for word, Vocab in emb_model.wv.vocab.items()]# 存储 所有的 词语

word_index = {" ": 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
word_vector = {} # 初始化`[word : vector]`字典

# 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
embedding_matrix = np.zeros((len(vocab_list) + 1, emb_model.vector_size))

for i in range(len(vocab_list)):
    # print(i)
    word = vocab_list[i]  # 每个词语
    word_index[word] = i + 1 # 词语：索引
    word_vector[word] = emb_model.wv[word] # 词语：词向量
    embedding_matrix[i + 1] = emb_model.wv[word]  # 词向量矩阵

print(embedding_matrix.shape)


# %%
result=[]
hit=0
miss=0
for row in tqdm(total_df[['user_id','ad_id_seq']].values,total=len(total_df)):
    try:
        result.append([row[0],[word_index[i]  for i in row[-1]]])
        hit+=1
    except Exception as e:
        miss+=1
print(f'hit:{hit}, miss:{miss}')

int_seq_df  = pd.DataFrame(result,columns=['user_id','ad_id_int_seq'])
print(int_seq_df)


# %%
train_df  = int_seq_df[int_seq_df.user_id <=720000]
valid_df = int_seq_df[int_seq_df.user_id > 720000]
print(train_df)

train_df = train_df.merge(label_df,on='user_id',how='left')
train_df['age'] =train_df['age'] -1
valid_df = valid_df.merge(label_df,on='user_id',how='left')
valid_df['age'] =valid_df['age'] -1

train_x = np.array(train_df[['ad_id_int_seq']].values[:,0])
train_y = train_df[['age']].values

valid_x = np.array(valid_df[['ad_id_int_seq']].values[:,0])
valid_y = valid_df[['age']].values

train_statics_df =  user_base_statics_df[user_base_statics_df.user_id<= 720000]
valid_statics_df =  user_base_statics_df[user_base_statics_df.user_id > 720000]
print(valid_statics_df)

before_one_hot =  train_y.reshape([-1,1])
enc = OneHotEncoder()
enc.fit(before_one_hot)

one_hoted_train_y  = enc.transform(before_one_hot).toarray()
print(one_hoted_train_y.shape)

before_one_hot =  valid_y.reshape([-1,1])
print(before_one_hot)
enc = OneHotEncoder()
enc.fit(before_one_hot)

one_hoted_valid_y  = enc.transform(before_one_hot).toarray()
print(one_hoted_valid_y.shape)

print(train_x)
print(len(train_x))
maxlen = 100
train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=maxlen)
valid_x = keras.preprocessing.sequence.pad_sequences(valid_x, maxlen=maxlen)
print(train_x)


# %%

embed_dim = 64  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer
vocab_size= 3027361
statics_features_size = 7


inputs_seq = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim,embedding_matrix)
x_seq = embedding_layer(inputs_seq)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x_seq = transformer_block(x_seq)
x_seq = layers.GlobalAveragePooling1D()(x_seq)
x_seq = layers.Dropout(0.1)(x_seq)
x_seq = layers.Dense(32, activation="relu")(x_seq)

x_seq_mode  = keras.Model(inputs=inputs_seq, outputs=x_seq)

inputs_statics = layers.Input(shape=(statics_features_size,))
combined = concatenate([x_seq_mode.output, inputs_statics])

z = Dense(64, activation="relu")(combined)
z = Dense(32, activation="relu")(z)

outputs = layers.Dense(10, activation="softmax")(z)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print(train_x.shape)
print(one_hoted_train_y)
print(valid_x.shape)
print(one_hoted_valid_y)

mc= keras.callbacks.ModelCheckpoint(
    './model/transform_checkpoint.hdf5',
    monitor='val_acc',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq=100
)
tfb =  keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq=1000)


callbacks = [mc]

model.fit([train_x,train_statics_df],
         one_hoted_train_y, 
        validation_data=([valid_x,valid_statics_df],one_hoted_valid_y),
        #validation_steps=10,
        #callbacks=callbacks,
        shuffle=True,
        epochs=10)

model.save('model/transform_end.h5')


