#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime

import eval

ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
NEW_USER_FILE = "cache/JData_User_New.csv"

ALL_ACTION_FILE = 'sub/raw_all_Action.csv'

MINI_USER_TRAIN = 'sub/mini_user_train.csv'
MINI_USER_TEST = 'sub/mini_user_test.csv'

MINI_ACT_TRAIN = 'sub/mini_act_train.csv'
MINI_ACT_TEST = 'sub/mini_act_test.csv'

USER_TRAIN = 'sub/user_train.csv'
USER_VALID = 'sub/user_valid.csv'
USER_TEST = 'sub/user_test.csv'

NEW_PRODUCT = 'sub/product.csv'
MINI_TRAIN_LABEL = 'sub/mini_train_label.csv'
MINI_TEST_LABEL =  'sub/mini_test_label.csv'

# Display format
pd.options.display.float_format = '{:,.3f}'.format

def simple_choose(group):
    gs = set(group['type'])
    if 2 in gs and 3 not in gs and 4 not in gs:
        group['label'] = 1
    else:
        group['label'] = 0
    return group[['sku_id', 'user_id', 'label']]

def get_data_by_date(data, date1, date2):
    '''
    Input:
        data1: (month, day)
    Output:
        data in [date1, date2]
    '''
    date1 = '2016-0' + str(date1[0]) + '-' + '{:0>2}'.format(date1[1]) + ' 00:00:00'
    date2 = '2016-0' + str(date2[0]) + '-' + '{:0>2}'.format(date2[1]) + ' 23:59:59'
    print  date1, date2
    res = data[data['time'] >= date1]
    res = res[res['time'] <= date2]
    return res

df_act = pd.read_csv(MINI_ACT_TRAIN)
df_act = df_act[['user_id','sku_id','type']]

res = df_act.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
res = res[res['label'] > 0]
#  将重复的用户－商品对丢弃
res = res.drop_duplicates()
label = pd.read_csv(MINI_TRAIN_LABEL)
eval.eval(res,label)

# 保存结果
res.to_csv('res/test.csv',columns=['user_id','sku_id'], index=False)
