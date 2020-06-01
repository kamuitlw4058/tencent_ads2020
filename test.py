import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
data_path = '/data/workspace/kimi/tencent_ads/2020/dataset'

label_df = pd.read_csv(f'{data_path}/train_preliminary/user.csv')

for fold_,(trn_idx,val_idx) in enumerate(folds.split(label_df,label_df)):
    print(f'fold_:{fold_}')
    print(f'trn_idx:{trn_idx}')
    print(f'val_idx:{val_idx}')
    label_df.loc[val_idx, 'fold'] = fold_


print(label_df)
label_df['age'] =label_df['age'].astype(int)
t =  label_df['age'].map(dict(zip([1,2,3],[2,3,4])))
#print(t)

#label_df['age'].map(dict(zip(label_df['age'].unique(), range(0, label_df['age'].unique()))))
#print(label_df)
print(label_df['age'].unique())

print(range(0, label_df['age'].unique()))