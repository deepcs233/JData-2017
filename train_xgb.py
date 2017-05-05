#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np


import eval
import genfeat
import xgboost as xgb

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
# Display format
pd.options.display.float_format = '{:,.3f}'.format


# raw
df_r, label_r, users_r = genfeat.get_train_data()
# test
df_t, label_t, users_t = genfeat.get_test_data()

# UnderSample
df_r, label_r = genfeat.underSample(df_r, label_r, prob = 0.028)


dtrain=xgb.DMatrix(df_r.values[:], label=label_r)
dtest=xgb.DMatrix(df_t.values[:], label=label_t)
param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 212
param['nthread'] = 4
param['eval_metric'] = "auc"
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train( plst, dtrain, num_round, evallist)
'''
sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                               sub_test_start_date, sub_test_end_date)
'''
def test(threshold = 0.5,verbose = True):
    dtest = xgb.DMatrix(df_t.values[:])
    y = bst.predict(dtest)
    pred = pd.concat([users_t,pd.DataFrame(y)],axis=1,ignore_index=False)
    pr = pred[pred[0]> threshold]
    del pr[0]
    yture = pd.concat([users_t, pd.DataFrame(label_t)], axis =1)
    yture = yture[yture['label']>0]
    pr = pr.drop_duplicates('user_id')
    return eval.eval(pr,yture, verbose)


def submit():
    df_s,users_s = genfeat.gen_submit_data(True)
    dtest = xgb.DMatrix(df_s.values[:])
    y = bst.predict(dtest)
    pred = pd.concat([users_s,pd.DataFrame(y)],axis=1,ignore_index=False)
    pred = pred[pred[0]>0.57]
    del pred[0]
    pred = pred.drop_duplicates('user_id')
    pred['user_id'] = pred['user_id'].astype(np.int32)
    pred.to_csv(RESULT_FILE, index=False)
    
