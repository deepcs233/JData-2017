#-*- coding: utf-8 -*-

from datetime import datetime
import math
import random

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import eval

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
TMP_ACT_SUBMIT = 'cache/act_submit.csv'
TMP_USER_SUBMIT = 'cache/user_submit.csv'

pd.describe_option("use_inf_as_null")

def simple_choose(group):
    gs = set(group['type'])
    if 2 in gs and 3 not in gs and 4 not in gs:
        group['lastady_addcart_label'] = 1
    else:
        group['lastady_addcart_label'] = 0
    return group[['sku_id', 'user_id', 'lastady_addcart_label']]


def get_data_by_date(data, start_date, end_date):
    '''
    Input:
        data1: (month, day)
    Output:
        data in [date1, date2]
    '''
    start_date = '2016-0' + str(start_date[0]) + '-' + '{:0>2}'.format(start_date[1]) \
                 + ' 00:00:00'
    end_date = '2016-0' + str(end_date[0]) + '-' + '{:0>2}'.format(end_date[1]) \
               + ' 23:59:59'

    res = data[data['time'] >= start_date]
    res = res[res['time'] <= end_date]
    return res

def get_accumulate_action_feat(data, start_date, end_date):
    actions = get_data_by_date(data, start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame

    start_date = '2016-0' + str(start_date[0]) + '-' + '{:0>2}'.format(start_date[1]) \
                 + ' 00:00:00'
    end_date = '2016-0' + str(end_date[0]) + '-' + '{:0>2}'.format(end_date[1]) \
               + ' 23:59:59'

    #近期行为按时间衰减
    actions['weights'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') \
                                             - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
    print (actions.head(10))
    actions['action_1'] = actions['action_1'] * actions['weights']
    actions['action_2'] = actions['action_2'] * actions['weights']
    actions['action_3'] = actions['action_3'] * actions['weights']
    actions['action_4'] = actions['action_4'] * actions['weights']
    actions['action_5'] = actions['action_5'] * actions['weights']
    actions['action_6'] = actions['action_6'] * actions['weights']
    del actions['type']
    del actions['time']
    del actions['weights']
    actions = actions.groupby(['user_id', 'sku_id','cate'], as_index=False).sum()

    return actions


def get_train_data():
    df = pd.read_csv(MINI_ACT_TRAIN )
    print 'ok'
    df = df[['user_id', 'sku_id', 'type', 'time', 'cate']]
    
    


    # 提取最后一天UI特征，是否添加购物车且没有买东西
    df_last = get_data_by_date(df, (4,10), (4,10))
    df_last = df_last.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
    df_last = df_last.drop_duplicates()
    
    df = get_accumulate_action_feat(df, (4,1), (4,10))
    # 添加cate_i
    cate =pd.get_dummies(df['cate'], prefix='cate')
    df = pd.concat([df, cate], axis=1)

    df = pd.merge(df, df_last, how='left', on=['user_id', 'sku_id'])

 
    del cate
    del df['cate']
    
    print 'ok'
    '''
    >>> df_act.columns
    Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
           u'action_4', u'action_5', u'action_6'],
          dtype='object')
    '''
    df_u = pd.read_csv(MINI_USER_TRAIN)
    df = pd.merge(df, df_u, how='left', on='user_id')
    print 'ok'
    df_p = pd.read_csv(ALL_PRODUCT_FILE)
    df = pd.merge(df, df_p, how='left', on='sku_id')
    print 'ok'
    df_l = pd.read_csv(MINI_TRAIN_LABEL)
    df_l['label'] = 1
    df = pd.merge(df, df_l, how='left', on=['user_id', 'sku_id'])
    print 'ok'

    df = df.fillna(0)
    df = df.replace(np.inf, 0)

    label = pd.DataFrame(df['label'].copy())
    users = df[['user_id', 'sku_id']].copy()

    del df['Unnamed: 0_x']   
    del df['Unnamed: 0_y']
    del df['Unnamed: 0']
    
    del df['user_id']
    del df['sku_id']
    del df['label']
    return df, label, users

def get_test_data():
    df = pd.read_csv(MINI_ACT_TEST )
    print 'ok'
    df = df[['user_id', 'sku_id', 'type', 'time', 'cate']]
    
    # 提取最后一天UI特征，是否添加购物车且没有买东西
    df_last = get_data_by_date(df, (4,10), (4,10))
    df_last = df_last.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
    df_last = df_last.drop_duplicates()
    
    df = get_accumulate_action_feat(df, (4,1), (4,10))
    # 添加cate_i
    cate =pd.get_dummies(df['cate'], prefix='cate')
    df = pd.concat([df, cate], axis=1)

    df = pd.merge(df, df_last, how='left', on=['user_id', 'sku_id'])


    del cate
    del df['cate']
    
    print 'ok'
    '''
    >>> df_act.columns
    Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
           u'action_4', u'action_5', u'action_6'],
          dtype='object')
    '''
    df_u = pd.read_csv(MINI_USER_TEST)
    df = pd.merge(df, df_u, how='left', on='user_id')
    print 'ok'
    df_p = pd.read_csv(ALL_PRODUCT_FILE)
    df = pd.merge(df, df_p, how='left', on='sku_id')
    print 'ok'
    df_l = pd.read_csv(MINI_TEST_LABEL)
    df_l['label'] = 1
    df = pd.merge(df, df_l, how='left', on=['user_id', 'sku_id'])
    print 'ok'
    df = df.fillna(0)
    df = df.replace(np.inf, 0)

    label = pd.DataFrame(df['label'].copy())
    users = df[['user_id', 'sku_id']].copy()
    del df['Unnamed: 0_x']   
    del df['Unnamed: 0_y']
    del df['Unnamed: 0']
    
    del df['user_id']
    del df['sku_id']
    del df['label']
    return df, label, users


    

def gen_submit_data():
    df = pd.read_csv(ALL_ACTION_FILE)
    print 'ok'
    df = df[['user_id', 'sku_id', 'type', 'time', 'cate']]
    
    # 提取最后一天UI特征，是否添加购物车且没有买东西
    df_last = get_data_by_date(df, (4,15), (4,15))
    df_last = df_last.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
    df_last = df_last.drop_duplicates()
    
    df = get_accumulate_action_feat(df, (4,10), (4,15))
    # 添加cate_i
    cate =pd.get_dummies(df['cate'], prefix='cate')
    df = pd.concat([df, cate], axis=1)

    df = pd.merge(df, df_last, how='left', on=['user_id', 'sku_id'])


    del cate
    del df['cate']
    print 'ok'
    '''
    >>> df_act.columns
    Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
           u'action_4', u'action_5', u'action_6'],
          dtype='object')
    '''
    df_u = pd.read_csv(ALL_USER_FILE)
    df = pd.merge(df, df_u, how='left', on='user_id')
    print 'ok'
    df_p = pd.read_csv(ALL_PRODUCT_FILE)
    df = pd.merge(df, df_p, how='left', on='sku_id')
    print 'ok'
    df = df.fillna(0)
    df = df.replace(np.inf, 0)


    users = df[['user_id', 'sku_id']].copy()
    del df['Unnamed: 0_x']   
    del df['Unnamed: 0_y']

    
    del df['user_id']
    del df['sku_id']

    return df,  users

    
def underSample(df, label, users, prob = 0.01):
    '''
    Type(label) : Series
    欠采样,负例被采样的可能性为0.01
    采样后 正：负 = 5：1
    '''
    label['prob'] = label['label'].map(lambda x: random.random() - x)
    df = pd.concat([df, label['prob']], axis=1)
    users = pd.concat([users, label['prob']], axis=1)
    
    df = df[df['prob'] < prob]
    label = label[label['prob'] < prob]
    users = users[users['prob'] < prob]
    
    del df['prob']
    del label['prob']
    del users['prob']
    return df, label, users


    
# raw
df_r, label_r, users_r = get_train_data()
# test
df_t, label_t, users_t = get_test_data()


#


params = {}


def train_predict(df_r, label_r, users_r, df_t, label_t, users_t, samprob = 0.02, params = {}):
    '''
    训练模型并给出测试结果
    '''
    df, label, users = underSample(df_r, label_r, users_r, prob = samprob)
    model = RandomForestClassifier(**params)#LogisticRegression()
    model.fit(df, label.values[:])
    '''
    pred = model.predict(df)
    pred = pd.concat([users, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0].dropna()
    answer = pd.concat([users, pd.DataFrame(label)], axis =1)
    answer = answer[answer['label'] > 0]
    pred = pred.drop_duplicates('user_id')
    print 'Train F1:', eval.eval(pred, answer)
   '''
    pred = model.predict(df_t)
    # print metrics.classification_report(label_t, pred)

    pred = pd.concat([users_t, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0]
    answer = pd.concat([users_t, pd.DataFrame(label_t)], axis =1)
    answer = answer[answer['label'] > 0]
    pred = pred.drop_duplicates('user_id')
    return eval.eval(pred, answer), model


maxscore = -1
for simp in [0.03]:

    for est_num in [100]:
        params['n_estimators'] = est_num

        sumscore = 0
        for i in range(10):
            try:
                score, _ = train_predict(df_r, label_r, users_r, df_t, label_t, \
                                         users_t, simp, params)
            except Exception,e:
                # print e
                score = 0
    
            sumscore += score
        
        score = sumscore / 5
        print simp, est_num, score
        if score > maxscore:
            maxscore = score


'''
score, model = train_predict(df_r, label_r, users_r, df_t, label_t, users_t, 0.05, \
                             {'n_estimators': 100})
'''
def genResult():
    # df_s ,users_s = gen_submit_data()
    df_s = pd.read_csv(TMP_ACT_SUBMIT)
    users_s = pd.read_csv(TMP_USER_SUBMIT)
    pred = model.predict(df_s)
    pred = pd.concat([users_s, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0]
    del pred[0]
    pred = pred.drop_duplicates('user_id')
    pred.to_csv(RESULT_FILE, index=False)

