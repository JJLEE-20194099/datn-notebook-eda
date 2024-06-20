import pandas as pd

df = pd.read_parquet('./data/data_df_hcm_v2.parquet')
target_df = pd.read_parquet('./data/target_df_hcm_v2.parquet')


df

import json

FS = json.load(open('./data/hcm_v2.json', 'r'))
cat_cols = FS['cat_cols']
num_cols = FS['num_cols']
all_cols = cat_cols + num_cols

df = df[all_cols]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import random

def create_model(cat_idxs = None):
    model = KNeighborsRegressor()
    search_params = {'n_neighbors': [1000 ,3000, 5000]}
    clf = GridSearchCV(model, search_params, scoring=['explained_variance', 'max_error', 'neg_root_mean_squared_error', 'r2', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error'], refit='neg_root_mean_squared_error', cv=5)
    return clf

model = create_model()

print("Start training:")

full_df = pd.concat([df, target_df['target']], axis = 1)

train_df = full_df.iloc[:-35000]
test_df = full_df.iloc[-35000:]

def train_test_split_by_col(train_df, test_df, X_cols, y_col):
    X_train, X_test, y_train, y_test = train_df[X_cols], test_df[X_cols], train_df[y_col], test_df[y_col]

    return X_train, X_test, y_train, y_test


target_feature = 'target'

X_train, X_test, y_train, y_test = train_test_split_by_col(train_df = train_df, test_df = test_df, X_cols = all_cols, y_col = target_feature)
model.fit(X_train, y_train)

print(model.predict(X_test)[:10])

from joblib import dump, load


import os

path = './model/hcm/knr/v2/'

os.makedirs(path, exist_ok=True)

estimator = model.best_estimator_
dump(estimator, f"{path}model.joblib")

load_model = load("./model/hcm/knr/v2/model.joblib")

print(load_model.predict(X_test)[:10])