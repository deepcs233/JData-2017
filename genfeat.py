#-*- coding: utf-8 -*-
from collections import Counter
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

TMP_MINI_TRAIN = 'cache/mini_train.csv'
TMP_MINI_TEST = 'cache/mini_test.csv'

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

    res = data[(data['time'] >= start_date) & (data['time']  <= end_date)]

    return res

def get_accumulate_action_feat(data, start_date, end_date):
    '''
    得到时间窗口内的action累计特征
    '''
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
    # rint (actions.head(10))
    actions['action_1'] = actions['action_1'] * actions['weights']
    actions['action_2'] = actions['action_2'] * actions['weights']
    actions['action_3'] = actions['action_3'] * actions['weights']
    actions['action_4'] = actions['action_4'] * actions['weights']
    actions['action_5'] = actions['action_5'] * actions['weights']
    actions['action_6'] = actions['action_6'] * actions['weights']
    del actions['type']
    del actions['time']
    del actions['weights']
    actions = actions.groupby(['user_id', 'sku_id','cate','brand'], as_index=False).sum()
    actions = actions[['user_id', 'sku_id','cate','brand','action_1', 'action_2', 'action_3', 'action_4', 'action_5','action_6']]

    return actions

def get_action_feat(data, start_date, end_date):

    actions = get_data_by_date(data, start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (str(start_date), str(end_date)))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del actions['type']

    return actions

def get_user_active(data, start_date, end_date):
    actions = get_data_by_date(data, start_date, end_date)
    actions = actions[['user_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-user_active' % (str(start_date), str(end_date)))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby(['user_id'], as_index=False).sum()

    return actions[['user_id','%s-%s-user_active_1' % (str(start_date), str(end_date)),
                    '%s-%s-user_active_4' % (str(start_date), str(end_date)),
                    '%s-%s-user_active_2' % (str(start_date), str(end_date)),]]

def get_product_active(data, start_date, end_date):
    actions = get_data_by_date(data, start_date, end_date)
    actions = actions[['sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-product_active' % (str(start_date), str(end_date)))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby(['sku_id'], as_index=False).sum()

    return actions[['sku_id','%s-%s-product_active_1' % (str(start_date), str(end_date)),
                    '%s-%s-product_active_4' % (str(start_date), str(end_date)),
                    '%s-%s-product_active_2' % (str(start_date), str(end_date)),]]


def get_cate_info(data):
    actions = data[['cate', 'type']]
    df = pd.get_dummies(actions['type'], prefix='cate_act')
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    del actions['type']
    actions = actions.groupby(['cate'], as_index=False).sum()

    return actions

def get_brand_info(data):
    actions = data[['brand', 'type']]
    df = pd.get_dummies(actions['type'], prefix='brand_act')
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    del actions['type']
    df_a = actions.groupby(['brand'], as_index=False).sum()

    #　构造转化率字段
    df_a['brand_buy_addcart_ratio'] = df_a['brand_act_4'] /  df_a['brand_act_2']
    df_a['brand_buy_delcart_ratio'] = df_a['brand_act_4'] / df_a['brand_act_3'] 
    df_a['brand_buy_browse_ratio'] = df_a['brand_act_4'] /  df_a['brand_act_1']
    df_a['brand_buy_click_ratio'] = df_a['brand_act_4'] /  df_a['brand_act_6']
    df_a['brand_buy_favor_ratio'] = df_a['brand_act_4'] /  df_a['brand_act_5']
    
    # 将大于１的转化率字段置为１(100%)
    df_a.ix[df_a['brand_buy_addcart_ratio'] > 1., 'brand_buy_addcart_ratio'] = 1.
    df_a.ix[df_a['brand_buy_delcart_ratio'] > 1., 'brand_del_addcart_ratio'] = 1.
    df_a.ix[df_a['brand_buy_browse_ratio'] > 1., 'brand_buy_browse_ratio'] = 1.
    df_a.ix[df_a['brand_buy_click_ratio'] > 1., 'brand_buy_click_ratio'] = 1.
    df_a.ix[df_a['brand_buy_favor_ratio'] > 1., 'brand_buy_favor_ratio'] = 1.
    
    return df_a

def get_user_mean_std(df,end_date=(4,15)):

    actions = df[['user_id', 'type','time']]
    df = pd.get_dummies(actions['type'], prefix='user_act')
    actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame


    end_date = '2016-0' + str(end_date[0]) + '-' + '{:0>2}'.format(end_date[1]) \
               + ' 23:59:59'

    #近期行为按时间衰减
    actions['days'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') \
                                             - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    actions['days'] = actions['days'].map(lambda x: x.days)
    del actions['type']
    del actions['time']
    actions = actions.groupby(['user_id','days'],as_index=False).sum()
    del actions['days']
    df_1 = actions.groupby('user_id', as_index=False).var()
    df_2 = actions.groupby('user_id', as_index=False).mean()
    df = pd.merge(df_2,df_1,how='left',on='user_id')
    df['user_id'] = df['user_id'].astype(np.int32)
    return df

def get_type_count(group):
    behavior_type = group.type.astype(int)
    
    # 用户行为类别

    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = np.sum(behavior_type ==6)

    return group[['user_id', 'sku_id', 'browse_num']]

def get_ui_num(group):
        group['ui_num'] = len(group)
        return group
    
def get_user_sku_rank(df):
    df = df[['user_id', 'sku_id', 'type']]

    def get_rank(group):
        num = len(group)
        group = group.sort_values('ui_num')
        group = group.reset_index()
        del group['index']
        del group['ui_num']
        group = pd.concat([group,pd.DataFrame(list(range(num)),columns=['user_sku_rank'])],
                          axis = 1)
        return group
    df = df.groupby(['user_id', 'sku_id'], as_index=False).apply(get_ui_num)
    del df['type']
    df = df.drop_duplicates(['user_id','sku_id'])
    df = df.groupby('user_id',as_index=False).apply(get_rank)
    df = df.reset_index()
    del df['level_0']
    del df['level_1']
    return df

def get_sku_rank(df):
    df = df[['sku_id', 'type']]
    
    df = df.groupby(['sku_id'], as_index=False).apply(get_ui_num)
    del df['type']
    df = df.drop_duplicates()
    num = len(df)
    df = df.sort_values('ui_num')
    df = df.reset_index()
    del df['index']
    del df['ui_num']
    df = pd.concat([df ,pd.DataFrame(list(range(num)),columns=['sku_rank'])],
                          axis = 1)
    return df

def get_cate_sku_rank(df):
    df = df[['cate', 'sku_id', 'type']]

    def get_rank(group):
        num = len(group)
        group = group.sort_values('ui_num')
        group = group.reset_index()
        del group['index']
        del group['ui_num']
        group = pd.concat([group,pd.DataFrame(list(range(num)),columns=['cate_sku_rank'])],
                          axis = 1)
        return group
    df = df.groupby(['cate', 'sku_id'], as_index=False).apply(get_ui_num)
    del df['type']
    df = df.drop_duplicates(['cate','sku_id'])
    df = df.groupby('cate',as_index=False).apply(get_rank)
    df = df.reset_index()
    del df['level_0']
    del df['level_1']
    del df['cate']
    return df
    return df

def get_train_data(reuse = False):

    if reuse:
        df = pd.read_csv(TMP_MINI_TRAIN)
    else:
        df = pd.read_csv(MINI_ACT_TRAIN)
        df = get_data_by_date(df, (3,25), (4,10))

        
        print ('load ok')
        brand_info = pd.read_csv('cache/train_brand_info.csv')
##        brand_info = get_brand_info(df)
##        brand_info.to_csv('cache/train_brand_info.csv',index=False)
        
        user_mean_std = pd.read_csv('cache/train_user_mean_std.csv')    
##        user_mean_std = get_user_mean_std(df)
##        user_mean_std.to_csv('cache/train_user_mean_std.csv',index=False)
        df = df[['user_id', 'sku_id', 'type', 'time', 'cate', 'brand']]

        df_last3 = get_data_by_date(df, (4,8), (4,10))

        # 提取每个user对sku的关注度排名
        user_sku_rank = pd.read_csv('cache/train_user_sku_rank.csv')
##        user_sku_rank = get_user_sku_rank(df_last3)
##        user_sku_rank.to_csv('cache/train_user_sku_rank.csv',index=False)

        # 提取sku被关注度排名
        sku_rank = get_sku_rank(df_last3)
        
        # 提取每个cate的sku的关注度排名
        cate_sku_rank = get_cate_sku_rank(df_last3)
        print('ok')
        # 提取最后一天UI特征，是否添加购物车且没有买东西
        df_last_two = pd.read_csv('cache/train_df_last_two.csv')
##        df_last_two = get_data_by_date(df, (4,9), (4,10))
##        df_last_two = df_last_two.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
##        df_last_two = df_last_two.drop_duplicates()
##        df_last_two.to_csv('cache/train_df_last_two.csv',index=False)
        df_last_one = pd.read_csv('cache/train_df_last_one.csv')
##        df_last_one = get_data_by_date(df, (4,10), (4,10))
##        df_last_one = df_last_one.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
##        df_last_one = df_last_one.drop_duplicates()
##        df_last_one.to_csv('cache/train_df_last_one.csv',index=False)
        
        print('ok')
        # 计算用户在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_user_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_user_active is None:
                df_user_active = get_user_active(df, (4,i), (4,10))
            else:
                df_user_active = pd.merge(df_user_active, get_user_active(df, (4,i), (4,10)), how='left',
                                          on=['user_id'])          
        df_user_active = df_user_active.fillna(0)
        
        # 计算商品在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_product_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_product_active is None:
                df_product_active = get_product_active(df, (4,i), (4,10))
            else:
                df_product_active = pd.merge(df_product_active, get_product_active(df, (4,i), (4,10)), how='left',
                                          on=['sku_id'])
        df_product_active = df_product_active.fillna(0)
        print('ok')
        # 计算cate-UI的交叉特征
        df_cate_info = get_cate_info(df)
        df_cate_info = df_cate_info.fillna(0)
        # 排序特征：@用户交互的商品中的排序
        #df_rank = df.groupby(['user_id'], as_index=False).apply(get_user_item_rank)
        #df_rank = df.drop_duplicates()


        
        # 创建用户在n天内与物品的交互特征
        df_act = None
        for start_date in [(4,1),(4,3),(4,6),(4,7),(4,8),(4,9),(4,10)]:
            if df_act is None:
                df_act = get_action_feat(df, start_date, (4,10))
            else:
                df_act = pd.merge(df_act, get_action_feat(df, start_date, (4,10)), how='left',
                                       on=['user_id', 'sku_id'])
        

        # 计算用户在n天内与物品的滑动特征，并完成合并
        '''
        df_1 = get_accumulate_action_feat(df, (4,3), (4,10))
        df_2 = get_accumulate_action_feat(df, (4,5), (4,10))
        df_3 = get_accumulate_action_feat(df, (4,7), (4,10))
        df = pd.merge(df_1, df_2, how='left', on=['user_id', 'sku_id', 'cate'])
        df = pd.merge(df, df_3, how='left', on=['user_id', 'sku_id', 'cate'])
        '''
        # df = pd.read_csv('cache/mini_train_accum_feat.csv')
        
        df = get_accumulate_action_feat(df, (4,5), (4,10))
        df.to_csv('cache/mini_train_accum_feat.csv', index=False)

        
        # 添加cate_i
        cate =pd.get_dummies(df['cate'], prefix='cate')
        df = pd.concat([df, cate], axis=1)

        df = pd.merge(df, df_last_one, how='left', on=['user_id', 'sku_id'])
        df = pd.merge(df, df_last_two, how='left', on=['user_id', 'sku_id'])
                
        df = pd.merge(df, df_user_active, how='left', on='user_id')
        df = pd.merge(df, df_product_active, how='left', on='sku_id')
        df = pd.merge(df, df_cate_info, how='left', on='cate')

        df = pd.merge(df, df_act, how='left', on=['user_id', 'sku_id'])

        df = pd.merge(df, cate_sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, user_sku_rank, how='left',on=['user_id','sku_id'])

        df = pd.merge(df, brand_info, how='left',on='brand')
        df = pd.merge(df, user_mean_std, how='left', on='user_id')
        
        del cate
        del df['cate']

        print ('ok')
        '''
        >>> df_act.columns
        Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
               u'action_4', u'action_5', u'action_6'],
              dtype='object')
        '''
        df_u = pd.read_csv(MINI_USER_TRAIN)
        df = pd.merge(df, df_u, how='left', on='user_id')

        df_p = pd.read_csv(ALL_PRODUCT_FILE)
        df = pd.merge(df, df_p, how='left', on='sku_id')

        df_l = pd.read_csv(MINI_TRAIN_LABEL)
        df_l['label'] = 1
        df = pd.merge(df, df_l, how='left', on=['user_id', 'sku_id'])
        print ('ok')

        df = df.fillna(0)
        df = df.replace(np.inf, 0)



        del df['Unnamed: 0_x']
        del df['Unnamed: 0_y']
        df = df.drop_duplicates()
        df.to_csv(TMP_MINI_TRAIN,index=False)
    
    df = df.drop_duplicates()
    label = pd.DataFrame(df['label'].copy())
    users = df[['user_id', 'sku_id']].copy()

    del df['user_id']
    del df['sku_id']
    del df['label']

    print('Success')

    return df, label, users

def get_test_data(reuse = False):
    if reuse:
        df = pd.read_csv(TMP_MINI_TEST)
    else:
        df = pd.read_csv(MINI_ACT_TEST)
        df = get_data_by_date(df, (3,25), (4,10))

        
        print ('ok')
        brand_info = get_brand_info(df)
        user_mean_std = get_user_mean_std(df)
        df = df[['user_id', 'sku_id', 'type', 'time', 'cate', 'brand']]

        df_last3 = get_data_by_date(df, (4,8), (4,10))

        # 提取每个user对sku的关注度排名
        user_sku_rank = get_user_sku_rank(df_last3)

        # 提取sku被关注度排名
        sku_rank = get_sku_rank(df_last3)
        
        # 提取每个cate的sku的关注度排名
        cate_sku_rank = get_cate_sku_rank(df_last3)
        
        # 提取最后一天UI特征，是否添加购物车且没有买东西
        df_last_two = get_data_by_date(df, (4,9), (4,10))
        df_last_two = df_last_two.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
        df_last_two = df_last_two.drop_duplicates()
        
        df_last_one = get_data_by_date(df, (4,10), (4,10))
        df_last_one = df_last_one.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
        df_last_one = df_last_one.drop_duplicates()
        # 计算用户在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_user_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_user_active is None:
                df_user_active = get_user_active(df, (4,i), (4,10))
            else:
                df_user_active = pd.merge(df_user_active, get_user_active(df, (4,i), (4,10)), how='left',
                                          on=['user_id'])          
        df_user_active = df_user_active.fillna(0)
        
        # 计算商品在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_product_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_product_active is None:
                df_product_active = get_product_active(df, (4,i), (4,10))
            else:
                df_product_active = pd.merge(df_product_active, get_product_active(df, (4,i), (4,10)), how='left',
                                          on=['sku_id'])
        df_product_active = df_product_active.fillna(0)

        # 计算cate-UI的交叉特征
        df_cate_info = get_cate_info(df)
        df_cate_info = df_cate_info.fillna(0)
        # 排序特征：@用户交互的商品中的排序
        #df_rank = df.groupby(['user_id'], as_index=False).apply(get_user_item_rank)
        #df_rank = df.drop_duplicates()


        
        # 创建用户在n天内与物品的交互特征
        df_act = None
        for start_date in [(4,1),(4,3),(4,6),(4,7),(4,8),(4,9),(4,10)]:
            if df_act is None:
                df_act = get_action_feat(df, start_date, (4,10))
            else:
                df_act = pd.merge(df_act, get_action_feat(df, start_date, (4,10)), how='left',
                                       on=['user_id', 'sku_id'])
        

        # 计算用户在n天内与物品的滑动特征，并完成合并
        '''
        df_1 = get_accumulate_action_feat(df, (4,3), (4,10))
        df_2 = get_accumulate_action_feat(df, (4,5), (4,10))
        df_3 = get_accumulate_action_feat(df, (4,7), (4,10))
        df = pd.merge(df_1, df_2, how='left', on=['user_id', 'sku_id', 'cate'])
        df = pd.merge(df, df_3, how='left', on=['user_id', 'sku_id', 'cate'])
        '''
        # df = pd.read_csv('cache/mini_test_accum_feat.csv')
        df = get_accumulate_action_feat(df, (4,5), (4,10))
        df.to_csv('cache/mini_test_accum_feat.csv', index=False)
        
        cate =pd.get_dummies(df['cate'], prefix='cate')
        df = pd.concat([df, cate], axis=1)

        df = pd.merge(df, df_last_one, how='left', on=['user_id', 'sku_id'])
        df = pd.merge(df, df_last_two, how='left', on=['user_id', 'sku_id'])
        df = pd.merge(df, df_user_active, how='left', on='user_id')
        df = pd.merge(df, df_product_active, how='left', on='sku_id')
        df = pd.merge(df, df_cate_info, how='left', on='cate')
        df = pd.merge(df, df_act, how='left', on=['user_id', 'sku_id'])

        df = pd.merge(df, cate_sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, user_sku_rank, how='left',on=['user_id','sku_id'])
        
        df = pd.merge(df, brand_info, how='left',on='brand')
        df = pd.merge(df, user_mean_std, how='left', on='user_id')
        del cate
        del df['cate']

        print ('ok')
        '''
        >>> df_act.columns
        Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
               u'action_4', u'action_5', u'action_6'],
              dtype='object')
        '''
        df_u = pd.read_csv(MINI_USER_TEST)
        df = pd.merge(df, df_u, how='left', on='user_id')

        df_p = pd.read_csv(ALL_PRODUCT_FILE)
        df = pd.merge(df, df_p, how='left', on='sku_id')
        print ('ok')
        df_l = pd.read_csv(MINI_TEST_LABEL)
        df_l['label'] = 1
        df = pd.merge(df, df_l, how='left', on=['user_id', 'sku_id'])
        print ('ok')

        df = df.fillna(0)
        df = df.replace(np.inf, 0)



        del df['Unnamed: 0_x']
        del df['Unnamed: 0_y']
        if 'Unnamed: 0' in df:
            del df['Unnamed: 0']
        df.to_csv(TMP_MINI_TEST, index=False)
    df = df.drop_duplicates()
    label = pd.DataFrame(df['label'].copy())
    users = df[['user_id', 'sku_id']].copy()
    
    del df['user_id']
    del df['sku_id']
    del df['label']

    return df, label, users



def gen_submit_data(reuse = False):
    if reuse:
        df = pd.read_csv(TMP_ACT_SUBMIT)
        users = pd.read_csv(TMP_USER_SUBMIT)
    else:

        df = pd.read_csv(ALL_ACTION_FILE)

        df = get_data_by_date(df, (4,1), (4,15))

        
        print ('ok')
        brand_info = get_brand_info(df)
##        user_mean_std = pd.read_csv('cache/submit_user_mean_std.csv')
##        user_mean_std = get_user_mean_std(df)
        user_mean_std.to_csv('cache/submit_user_mean_std.csv',index=False)
        df = df[['user_id', 'sku_id', 'type', 'time', 'cate', 'brand']]
        
        df_last3 = get_data_by_date(df, (4,8), (4,10))

        # 提取每个user对sku的关注度排名
        # user_sku_rank = pd.read_csv('cache/submit_user_sku_rank.csv')
        user_sku_rank = get_user_sku_rank(df_last3)
        user_sku_rank.to_csv('cache/submit_user_sku_rank.csv',index=False)

        # 提取sku被关注度排名
        sku_rank = get_sku_rank(df_last3)
        
        # 提取每个cate的sku的关注度排名
        cate_sku_rank = get_cate_sku_rank(df_last3)
        print('ok')
        
        # 提取最后一天UI特征，是否添加购物车且没有买东西
        df_last_two = pd.read_csv('cache/submit_last_two.csv')
##        df_last_two = get_data_by_date(df, (4,14), (4,15))
##        df_last_two = df_last_two.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
##        df_last_two = df_last_two.drop_duplicates()
##        df_last_two.to_csv('cache/submit_last_two.csv',index=False)
        print('ok')
        
        df_last_one = pd.read_csv('cache/submit_last_one.csv')
##        df_last_one = get_data_by_date(df, (4,14), (4,15))
##        df_last_one = df_last_one.groupby(['user_id','sku_id'], as_index=False).apply(simple_choose)
##        df_last_one = df_last_one.drop_duplicates()
##        df_last_two.to_csv('cache/submit_last_one.csv',index=False)
        print('ok')
        # 计算用户在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_user_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_user_active is None:
                df_user_active = get_user_active(df, (4,i+5), (4,15))
            else:
                df_user_active = pd.merge(df_user_active, get_user_active(df, (4,i+5), (4,15)), how='left',
                                          on=['user_id'])          
        df_user_active = df_user_active.fillna(0)
        print('ok')
        # 计算商品在最后1,2,3,5天的活跃度特征（以浏览量计）
        df_product_active = None
        for i in [1,3,5,6,8,9,10]:
            if df_product_active is None:
                df_product_active = get_product_active(df, (4,i+5), (4,15))
            else:
                df_product_active = pd.merge(df_product_active, get_product_active(df, (4,i+5), (4,15)), how='left',
                                          on=['sku_id'])
        df_product_active = df_product_active.fillna(0)

        # 计算cate-UI的交叉特征
        df_cate_info = get_cate_info(df)
        df_cate_info = df_cate_info.fillna(0)
        # 排序特征：@用户交互的商品中的排序
        #df_rank = df.groupby(['user_id'], as_index=False).apply(get_user_item_rank)
        #df_rank = df.drop_duplicates()


        
        # 创建用户在n天内与物品的交互特征
        df_act = None
        for start_date in [(4,6),(4,9),(4,11),(4,12),(4,13),(4,14),(4,15)]:
            if df_act is None:
                df_act = get_action_feat(df, start_date, (4,15))
            else:
                df_act = pd.merge(df_act, get_action_feat(df, start_date, (4,15)), how='left',
                                       on=['user_id', 'sku_id'])
        

        # 计算用户在n天内与物品的滑动特征，并完成合并
        '''
        df_1 = get_accumulate_action_feat(df, (4,8), (4,15))
        df_2 = get_accumulate_action_feat(df, (4,10), (4,15))
        df_3 = get_accumulate_action_feat(df, (4,12), (4,15))
        df = pd.merge(df_1, df_2, how='left', on=['user_id', 'sku_id', 'cate'])
        df = pd.merge(df, df_3, how='left', on=['user_id', 'sku_id', 'cate'])
        '''
        df = pd.read_csv('cache/submit_accumu.csv')
##        df = get_accumulate_action_feat(df, (4,10), (4,15))
##        df.to_csv('cache/submit_accumu.csv',index=False)
        # 添加cate_i
        cate =pd.get_dummies(df['cate'], prefix='cate')
        df = pd.concat([df, cate], axis=1)

        df = pd.merge(df, df_last_one, how='left', on=['user_id', 'sku_id'])
        df = pd.merge(df, df_last_two, how='left', on=['user_id', 'sku_id'])
        df = pd.merge(df, df_user_active, how='left', on='user_id')
        df = pd.merge(df, df_product_active, how='left', on='sku_id')
        df = pd.merge(df, df_cate_info, how='left', on='cate')
        df = pd.merge(df, df_act, how='left', on=['user_id', 'sku_id'])

        df = pd.merge(df, cate_sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, sku_rank, how='left',on=['sku_id'])
        df = pd.merge(df, user_sku_rank, how='left',on=['user_id','sku_id'])

        df = pd.merge(df, brand_info, how='left',on='brand')
        df = pd.merge(df, user_mean_std, how='left', on='user_id')
        
        del cate
        del df['cate']

        print ('ok')
        '''
        >>> df_act.columns
        Index([u'user_id', u'sku_id', u'action_1', u'action_2', u'action_3',
               u'action_4', u'action_5', u'action_6'],
              dtype='object')
        '''
        df_u = pd.read_csv(ALL_USER_FILE)
        df = pd.merge(df, df_u, how='left', on='user_id')
        print ('ok')
        df_p = pd.read_csv(ALL_PRODUCT_FILE)
        df = pd.merge(df, df_p, how='left', on='sku_id')
        print ('ok')



        df = df.fillna(0)
        df = df.replace(np.inf, 0)



        del df['Unnamed: 0_x']
        del df['Unnamed: 0_y']

        df = df.drop_duplicates()
        users = df[['user_id', 'sku_id']].copy()

        del df['user_id']
        del df['sku_id']

        
        
        df.to_csv(TMP_ACT_SUBMIT, index=False)
        users.to_csv(TMP_USER_SUBMIT, index=False)

    return df,  users


def underSample(df, label, prob = 0.01):
    '''
    Type(label) : Series
    欠采样,负例被采样的可能性为0.01
    采样后 正：负 = 5：1
    '''
    label['prob'] = label['label'].map(lambda x: random.random() - x)
    df = pd.concat([df, label['prob']], axis=1)


    df = df[df['prob'] < prob]
    label = label[label['prob'] < prob]


    del df['prob']
    del label['prob']

    return df, label






def train_predict(df_r, label_r, users_r, df_t, label_t, users_t, samprob = 0.02, params = {}):
    '''
    训练模型并给出测试结果
    '''
    score = {}
    
    df, label = underSample(df_r, label_r, prob = samprob)
    model = RandomForestClassifier(**params)#RandomForestClassifier(**params)#LogisticRegression()
    model.fit(df.values[:], label.values[:])
 
    pred = model.predict(df_r)
    pred = pd.concat([users_r, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0]
    answer = pd.concat([users_r, pd.DataFrame(label)], axis =1)
    answer = answer[answer['label'] > 0]
    pred = pred.drop_duplicates('user_id')

    
    score['train'] = eval.eval(pred, answer)    
    
    pred = model.predict(df_t)
    # print metrics.classification_report(label_t, pred)

    pred = pd.concat([users_t, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0]
    answer = pd.concat([users_t, pd.DataFrame(label_t)], axis =1)
    answer = answer[answer['label'] > 0]
    pred = pred.drop_duplicates('user_id')
    
    score['test'] = eval.eval(pred, answer)
    return score, model

def genResult():
    # df_s ,users_s = gen_submit_data()
    df_s = pd.read_csv(TMP_ACT_SUBMIT)
    users_s = pd.read_csv(TMP_USER_SUBMIT)
    pred = model.predict(df_s)
    pred = pd.concat([users_s, pd.DataFrame(pred)], axis =1)
    pred = pred[pred[0] > 0]
    del pred[0]

    df_p = pd.read_csv(PRODUCT_FILE)
    p = set(df_p['sku_id'])
    pred['Chosed'] = pred['sku_id'].map(lambda x: x in p)
    pred = pred[pred['Chosed'] == True]
    del pred['Chosed']

    pred = pred.drop_duplicates('user_id')
    pred['user_id'] = pred['user_id'].astype(np.int32)
    pred.to_csv(RESULT_FILE, index=False)

if __name__ == '__main__':

    print ('Start')
    # raw
    df_r, label_r, users_r = get_train_data(True)
    # test
    df_t, label_t, users_t = get_test_data(True)




    params = {}
    maxscore = -1
    for simp in [0.025,0.03,0.035]          :
        for est_num in [80,100,120]:
            for depth in [100]:
                params['n_estimators'] = est_num
                # params['max_depth'] = depth
                sum_train_score = 0
                sum_test_score = 0
                for i in range(10):

                    score, _ = train_predict(df_r, label_r, users_r, df_t, label_t, \
                                                 users_t, simp, params)

                    sum_train_score += score['train']
                    sum_test_score += score['test']

                sum_train_score = sum_train_score / 10
                sum_test_score = sum_test_score / 10
                print (simp, est_num,depth, 'Train:',sum_train_score,'Test:',sum_test_score)
                if score > maxscore:
                    maxscore = score


'''
score, model = train_predict(df_r, label_r, users_r, df_t, label_t, users_t, 0.026, \
                             {'n_estimators': 100, 'max_depth' : 125})
'''



