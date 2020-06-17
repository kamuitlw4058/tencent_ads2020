# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import random
import gc
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
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
train_df =pd.read_pickle('train5.pkl')
train_df['age']  = train_df['age'] -1

valid_df =pd.read_pickle('valid5.pkl')
valid_df['age']  = valid_df['age'] -1

user_statics_df =pd.read_pickle(f'{preprocess_path}/user_statics_train.pkl').drop(['click_times_total','active_days'],axis=1)

train_df = train_df.merge(user_statics_df,on='user_id')
valid_df = valid_df.merge(user_statics_df,on='user_id')

print(train_df)
print(valid_df)


# %%
final_train_x_df = train_df.drop(['age','user_id','gender','advertiser_id_seq','industry_seq'], axis=1)
#final_train_x_df = train_df.drop(['age','user_id','gender','active_days'], axis=1)
final_train_y_df = train_df['age']

final_valid_x_df = valid_df.drop(['age','user_id','gender','advertiser_id_seq','industry_seq'], axis=1)
final_valid_y_df = valid_df['age']
num_normal_features = ['_clicks_max_click_cnt','_max_clicked_ratio','_clicks_min_click_cnt','_min_clicked_ratio','_clicks_len','_clicks_mean','_clicks_median','_clicks_std']
num_date_features  = [ '_clicks_max_click_cnt', '_clicks_min_click_cnt','_clicks_len','_clicks_mean','_clicks_median','_clicks_std']
num_features = ['click_times_total'] +                [f'date{i}'  for i in num_date_features] +                 [f'wday{i}'  for i in num_date_features] +                 [f'month{i}'  for i in num_date_features] +                  [f'product_id{i}'  for i in num_normal_features] +                  [f'product_category{i}'  for i in num_normal_features] +                 [f'industry{i}'  for i in num_normal_features] +                 [f'advertiser_id{i}'  for i in num_normal_features]

#print(num_features)

c_features = ['industry_clicks_max_click','industry_clicks_min_click',
              'advertiser_id_clicks_max_click','advertiser_id_clicks_min_click',
              'product_id_clicks_max_click','product_id_clicks_min_click',
              'product_category_clicks_max_click','product_category_clicks_min_click',
             ]
features= num_features + c_features
topN = 3
def forfor(a): 
    return [item for sublist in a for item in sublist] 
features= ['active_days','click_times_total'] +              [f"industry_{i}" for i in range(16)] +             [f"advertiser_id_{i}" for i in range(32)]  +            forfor([[f'industry_top{i}_{j}'  for j in range(16)]  for i in range(topN)]) +             forfor([[f'advertiser_id_top{i}_{j}'  for j in range(32)]  for i in range(topN)]) +            'active_days_max,active_days_min,active_days_mean,active_days_std,week_active_days_max,week_active_days_min,week_active_days_mean,week_active_days_std,month_acitve_days_max,month_acitve_days_min,month_acitve_days_mean,month_acitve_days_std'.split(',')
#print(features)
print(len(features))

drop_features= [f"industry_{i}" for i in range(16)] +             [f"advertiser_id_{i}" for i in range(32)]  +            forfor([[f'industry_top{i}_{j}'  for j in range(16)]  for i in range(topN)]) +             forfor([[f'advertiser_id_top{i}_{j}'  for j in range(32)]  for i in range(topN)])
final_train_x_df = final_train_x_df.drop(drop_features,axis=1)
final_valid_x_df = final_valid_x_df.drop(drop_features,axis=1)
print(final_train_x_df)
final_train_y_one_hot_df =  final_train_y_df.values.reshape([-1,1])
enc = OneHotEncoder()
enc.fit(final_train_y_one_hot_df)

final_train_y_one_hot_df  = enc.transform(final_train_y_one_hot_df).toarray()
print(final_train_y_one_hot_df.shape)


final_valid_y_one_hot_df =  final_valid_y_df.values.reshape([-1,1])
enc = OneHotEncoder()
enc.fit(final_valid_y_one_hot_df)

final_valid_y_one_hot_df  = enc.transform(final_valid_y_one_hot_df).toarray()
print(final_valid_y_one_hot_df.shape)


# %%
import keras as K
    # 2. 定义模型
init = K.initializers.random_uniform()
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.normalization.BatchNormalization(axis=-1))
model.add(K.layers.Dense(units=64, input_dim=14,kernel_initializer='random_uniform', activation='relu'))
model.add(K.layers.Dense(units=32, kernel_initializer='random_uniform',activation='relu'))
model.add(K.layers.Dense(units=10, kernel_initializer='random_uniform',activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])


# %%
max_epochs = 10
print("Starting training ")
h = model.fit(final_train_x_df.values, final_train_y_one_hot_df,validation_data=(final_valid_x_df.values,final_valid_y_one_hot_df), batch_size=256, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")


# %%
y_pred = gbm.predict(final_train_x_df)
for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0
                
print(precision_score(one_hoted_y, y_pred,average='micro'))

ret = []
for user_id,age in zip(range(1000000),y_pred):
    ret.append([int(user_id),int(age.tolist().index(1) + 1)])
ret_df = pd.DataFrame(ret,columns=['user_id','predicted_age'])
print(ret_df['predicted_age'].value_counts())


# %%

before_one_hot =  final_valid_y_df.values.reshape([-1,1])
print(before_one_hot)
enc = OneHotEncoder()
enc.fit(before_one_hot)

one_hoted_y  = enc.transform(before_one_hot).toarray()
print(one_hoted_y.shape)


# %%
y_pred = gbm.predict(final_valid_x_df)
for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0
                
precision_score(one_hoted_y, y_pred,average='micro')


# %%
ret = []
for user_id,age in zip(range(1000000),y_pred):
    ret.append([int(user_id),int(age.tolist().index(1) + 1)])
ret_df = pd.DataFrame(ret,columns=['user_id','predicted_age'])
print(ret_df['predicted_age'].value_counts())


# %%



