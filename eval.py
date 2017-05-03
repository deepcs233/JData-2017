#encoding=utf-8
import numpy as np

def eval(pred, label, verbose = False):
    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    if (pos + neg) == 0:
        all_user_acc = 0
    else:
        all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)


    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
            
    if (pos + neg) == 0:
        all_item_acc = 0
    else:
        all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)


    if (all_user_recall + all_user_acc) == 0:
        F11 = 0
        F12 = 0
        score = 0
    else:
        F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
        F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
        score = 0.4 * F11 + 0.6 * F12

    if verbose:
        print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
        print ('所有用户中预测购买用户的召回率' + str(all_user_recall))
        print ('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
        print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
        print ('F11=' + str(F11))
        print ('F12=' + str(F12))
        print ('score=' + str(score))
    return score
