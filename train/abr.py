import pandas as pd
import json
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import random
import numpy as np
from joblib import dump, load
import os
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import random

def rmse(y_pred_test, y_test):
    return mean_squared_error(y_test, y_pred_test, squared = False)


def create_model(cat_idxs = None):
    model = AdaBoostRegressor(random_state=random.randint(3, 1000), n_estimators=12000, learning_rate=0.01)
    return model

def train_test_split_by_col(train_df, test_df, X_cols, y_col):
    X_train, X_test, y_train, y_test = train_df[X_cols], test_df[X_cols], train_df[y_col], test_df[y_col]

    return X_train, X_test, y_train, y_test

for version in tqdm([0, 1, 2, 3, 4]):
    for city in tqdm(['hcm', 'hn']):



        df = pd.read_parquet(f'../data/data_df_{city}_v{version}.parquet')
        target_df = pd.read_parquet(f'../data/target_df_{city}_v{version}.parquet')

        FS = json.load(open(f'../data/{city}_v{version}.json', 'r'))
        cat_cols = FS['cat_cols']
        num_cols = FS['num_cols']
        all_cols = cat_cols + num_cols

        df = df[all_cols]

        categorical_features_indices = [i for i, c in enumerate(all_cols) if c in cat_cols]
        model = create_model(categorical_features_indices)

        print("Start training:")

        target_df['target'] = target_df['target']

        full_df = pd.concat([df, target_df['target']], axis = 1)

        print(full_df.shape)

        sample_dict = {
            "hcm": 35000,
            "hn": 100000
        }

        sample = sample_dict[city]
        train_df = full_df.iloc[:sample]
        test_df = full_df.iloc[sample:]

        target_feature = 'target'

        X_train, X_test, y_train, y_test = train_test_split_by_col(train_df = train_df, test_df = test_df, X_cols = all_cols, y_col = target_feature)

        print(X_train.shape, X_test.shape)

        model.fit(X_train, y_train)


        y_pred_test = model.predict(X_test)
        print(y_pred_test[:10])

        path = f'../model/{city}/abr/v{version}/'

        os.makedirs(path, exist_ok=True)

        estimator = model
        dump(estimator, f"{path}model.joblib")

        load_model = load(f"../model/{city}/abr/v{version}/model.joblib")

        print(load_model.predict(X_test)[:10])

        pred_df = pd.DataFrame()
        pred_df['pred'] = y_pred_test
        pred_df['target'] = list(y_test)

        print("RMSE:", rmse(y_pred_test, list(y_test)))