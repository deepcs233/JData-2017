#-*- coding: utf-8 -*-
from collections import Counter
import pickle

import pandas as pd
import numpy as np

ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
NEW_USER_FILE = "cache/JData_User_New.csv"

ALL_ACTION_FILE = 'sub/raw_all_Action.csv'
ALL_PRODUCT_FILE = 'sub/all_product.csv'
ALL_USER_FILE = 'sub/all_user.csv'

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

RESULT_FILE = 'res/result.csv'

# Display format
pd.options.display.float_format = '{:,.3f}'.format

df = pd.read_csv(NEW_USER_FILE)

nums = len(df)


def convert_age(age_str):
    if age_str == u'-1':
        return -1
    elif age_str == u'15岁以下':
        return 0
    elif age_str == u'16-25岁':
        return 1
    elif age_str == u'26-35岁':
        return 2
    elif age_str == u'36-45岁':
        return 3
    elif age_str == u'46-55岁':
        return 4
    elif age_str == u'56岁以上':
        return 5
    else:
        return -1

def tranform_user_age(df):
    # Load data, header=0 means that the file has column names
    df = pd.read_csv(USER_FILE, header=0, encoding="gbk")


    df['age'] = df['age'].map(convert_age)
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'])
    min_date = min(df['user_reg_tm'])

    df['user_reg_diff'] = [i for i in (df['user_reg_tm'] - min_date).dt.days]
    del df['user_reg_tm']
    return df

# 功能函数: 对每一个user分组的数据进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['user_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

def p_add_type_count(group):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

# 将各个action数据的统计量进行聚合
def get_user_behavior(df_act):

    # 用户在不同action表中统计量求和
    df_a = df_act.groupby(['user_id'], as_index=False).apply(add_type_count)
    df_a = df_a.drop_duplicates('user_id')

    df_a = df_a.groupby(['user_id'], as_index=False).sum()
    #　构造转化率字段
    df_a['buy_addcart_ratio'] = df_a['buy_num'] /  df_a['addcart_num']
    df_a['buy_delcart_ratio'] = df_a['buy_num'] / df_a['delcart_num']  
    df_a['buy_browse_ratio'] = df_a['buy_num'] /  df_a['browse_num']
    df_a['buy_click_ratio'] = df_a['buy_num'] /  df_a['click_num']
    df_a['buy_favor_ratio'] = df_a['buy_num'] /  df_a['favor_num']
    
    # 将大于１的转化率字段置为１(100%)
    df_a.ix[df_a['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_a.ix[df_a['buy_delcart_ratio'] > 1., 'del_addcart_ratio'] = 1.
    df_a.ix[df_a['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_a.ix[df_a['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_a.ix[df_a['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_a

def get_product_behavior(df_act):

    # 用户在不同action表中统计量求和
    df_a = df_act.groupby(['sku_id'], as_index=False).apply(p_add_type_count)
    df_a = df_a.drop_duplicates('sku_id')

    df_a = df_a.groupby(['sku_id'], as_index=False).sum()
    
    #　构造转化率字段
    df_a['p_buy_addcart_ratio'] = df_a['buy_num'] /  df_a['addcart_num']
    df_a['p_buy_delcart_ratio'] = df_a['buy_num'] / df_a['delcart_num']  
    df_a['p_buy_browse_ratio'] = df_a['buy_num'] /  df_a['browse_num']
    df_a['p_buy_click_ratio'] = df_a['buy_num'] /  df_a['click_num']
    df_a['p_buy_favor_ratio'] = df_a['buy_num'] /  df_a['favor_num']
    
    # 将大于１的转化率字段置为１(100%)
    df_a.ix[df_a['p_buy_addcart_ratio'] > 1., 'p_buy_addcart_ratio'] = 1.
    df_a.ix[df_a['p_buy_delcart_ratio'] > 1., 'p_del_addcart_ratio'] = 1.
    df_a.ix[df_a['p_buy_browse_ratio'] > 1., 'p_buy_browse_ratio'] = 1.
    df_a.ix[df_a['p_buy_click_ratio'] > 1., 'p_buy_click_ratio'] = 1.
    df_a.ix[df_a['p_buy_favor_ratio'] > 1., 'p_buy_favor_ratio'] = 1.

    return df_a


with open('cache/users_set.pkl','r') as f:
    clean_users_set = pickle.load(f)

MINI_TRAIN_NUM = int(nums * 0.8)
MINI_TEST_NUM = int(nums * 1)
'''
TRAIN_NUM = int(nums * 0.6)
VALID_NUM = int(nums * 0.8)
TEST_NUM = int(nums * 1)

'''


df[: MINI_TRAIN_NUM].to_csv(MINI_USER_TRAIN, index= False)
df[MINI_TRAIN_NUM: MINI_TEST_NUM].to_csv(MINI_USER_TEST, index=False)
'''
df[: TRAIN_NUM].to_csv(USER_TRAIN, index=False)
df[TRAIN_NUM: VALID_NUM].to_csv(USER_VALID, index=False)
df[VALID_NUM: TEST_NUM].to_csv(USER_TEST, index=False)

'''

mini_user_train_set = set(df[: MINI_TRAIN_NUM]['user_id']) #- clean_users_set
mini_user_test_set = set(df[MINI_TRAIN_NUM: MINI_TEST_NUM]['user_id']) #- clean_users_set

acts = []
print ('ok')

# 使用重复变量节约内存
def getAction():
    df_act = pd.read_csv("data/JData_Action_201602.csv")
    acts.append(df_act)

    df_act = pd.read_csv("data/JData_Action_201603.csv")
    acts.append(df_act)

    df_act = pd.read_csv("data/JData_Action_201604.csv")
    acts.append(df_act)

    df_act = pd.concat(acts, ignore_index=True)
    return df_act
'''
df_act = getAction()
df_act.to_csv(ALL_ACTION_FILE, index=False)
'''
'''
# 计算product的各项特征
df_product_behavior = get_product_behavior(df_act)

'''
df_act = pd.read_csv(ALL_ACTION_FILE)
print ('ok')
df_act['user_id'] = df_act['user_id'].astype(np.int32)

# df_act.to_csv(ALL_ACTION_FILE)

df_act['Chosed'] = df_act['user_id'].map(lambda x: x in mini_user_train_set )
acts_train = df_act[df_act['Chosed'] == True]
del acts_train['Chosed']
print ('ok')


df_act['Chosed'] = df_act['user_id'].map(lambda x: x in mini_user_test_set)
acts_test = df_act[df_act['Chosed'] == True]
del acts_test['Chosed']
print ('ok')



df_act_train_label = acts_train[acts_train['time'] > '2016-04-10 23:59:59']

df_act = acts_train[acts_train['time'] < '2016-04-10 23:59:59']
df_user_train_behavior = get_user_behavior(df_act)
df_act.to_csv(MINI_ACT_TRAIN, index=False)

print ('ok')

df_act_test_label = acts_test[acts_test['time'] > '2016-04-10 23:59:59']

df_act = acts_test[acts_test['time'] < '2016-04-10 23:59:59']
df_user_test_behavior = get_user_behavior(df_act)
df_act.to_csv(MINI_ACT_TEST, index=False)
print ('ok')


del acts_test
del acts_train

# 将user与behavior拼接
df_u = pd.read_csv(MINI_USER_TRAIN)
df_u = pd.merge(df_u, df_user_train_behavior, how='inner')
df_u.to_csv(MINI_USER_TRAIN, ignore_index=True)
print ('ok')
df_u = pd.read_csv(MINI_USER_TEST)
df_u = pd.merge(df_u, df_user_test_behavior, how='inner')
df_u.to_csv(MINI_USER_TEST, ignore_index=True)

'''
print ('ok')
# 将product的基本信息,评论,与UI拼接
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"

df_p = pd.read_csv(PRODUCT_FILE)
df_c = pd.read_csv(COMMENT_FILE)

df_c = df_c.groupby(['sku_id'], as_index=False).mean()
df_cp = pd.merge(df_p,df_c, how='inner')
df_cp = pd.merge(df_cp, df_product_behavior, how='inner')
df_cp.to_csv(NEW_PRODUCT, ignore_index=True)
'''

print ('ok')
# 生成Train,与Test 的Labels

df_act = df_act_train_label
df_act = df_act[df_act['type'] == 4]
df_act = df_act[df_act['cate'] == 8][['user_id','sku_id']]
df_act.to_csv(MINI_TRAIN_LABEL, ignore_index=True)

df_act = df_act_test_label
df_act = df_act[df_act['type'] == 4]
df_act = df_act[df_act['cate'] == 8][['user_id','sku_id']]
df_act.to_csv(MINI_TEST_LABEL, ignore_index=True)

print ('ok')



