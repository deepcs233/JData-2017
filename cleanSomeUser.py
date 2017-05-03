#-*- coding: utf-8 -*-

import pickle

import pandas as pd
import numpy as np

ALL_USER_FILE = 'sub/all_user.csv'

df = pd.read_csv(ALL_USER_FILE)



df = df[(df['buy_num'] <= 0) & (df['addcart_num'] <= 0)]

user_set = set(df['user_id'])

with open('cache/users_set.pkl','w') as f:
    pickle.dump(user_set, f)
