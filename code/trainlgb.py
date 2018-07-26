# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:16:30 2018

@author: cjzhao_13
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:41:57 2018

@author: cjzhao_13
"""
def mytrainlgb():
    
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings("ignore")
    
    import time
    import pandas as pd

    
 
    train=pd.read_csv('../data/train.csv',encoding='gbk')
    test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',encoding='gbk')
    testt=pd.read_csv('../data/test.csv',encoding='gbk')
    
    print('haha====================================================')
    # =============================================================================
    # print(train.columns)
    # train=labelencodeall(train)
    # train.to_csv('../tmp/train.csv')
    # =============================================================================
    # =============================================================================
    # train=pd.merge(train,targetall,on='vid',how='left')
    # train=train[train.收缩压.notnull()]
    # testt=train[train.血清高密度脂蛋白.isnull()]
    # =============================================================================
    # =============================================================================
    # data=pd.concat([train,testt])
    # data=zuhe(data)
    # train=data[data.收缩压.notnull()]
    # testt=data[data.收缩压.isnull()]
    # =============================================================================
    print('============================')
    print(train.shape)
    print(test.shape)
    print(testt.shape)
    print('============================')
    train['收缩压'] = train['收缩压'].astype(float)
    train['舒张压'] =train['舒张压'].astype(float)
    train['血清甘油三酯']=train['血清甘油三酯'].astype(float)
    train['收缩压']=np.log(train['收缩压']+1)
    train['舒张压']=np.log(train['舒张压']+1)
    train['血清甘油三酯']=np.log(train['血清甘油三酯']-0.099999999)
    train['血清高密度脂蛋白']=np.log(train['血清高密度脂蛋白'])
    train['血清低密度脂蛋白']=np.log2(train['血清低密度脂蛋白']+1.22001)
    print(train.info())
    train.pop('vid')
    test_index=test.pop('vid')
    testt_index=testt.pop('vid')
    columns=list(train.columns)
    columns.remove('收缩压')
    columns.remove('舒张压')
    columns.remove('血清甘油三酯')
    columns.remove('血清高密度脂蛋白')
    columns.remove('血清低密度脂蛋白')
    print(train.mean())
    
    new=pd.DataFrame()
    new['vid']=testt_index
    col=['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    
    for t in col:
        print(t)
        X=train[columns]
        y=train[t]
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
        gbm = lgb.LGBMRegressor(objective='regression',
                               # num_leaves=23,
                                learning_rate=0.08,
                                #max_depth=25,
                                #max_bin=10000,
                                drop_rate=0.10,
                                
                                #is_unbalance=True,
                                n_estimators=1000)#1000
        gbm.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='l1',
                early_stopping_rounds=100)
        
        y_pred = gbm.predict(testt[columns], num_iteration=gbm.best_iteration)
        print('======================================================================================')
        predictors = [i for i in X_train.columns]
        feat_imp = pd.Series(gbm.feature_importance(), predictors).sort_values(ascending=False)
        print(feat_imp)
        new[t]=y_pred
    
    new['收缩压']=np.exp(new['收缩压'])-1
    new['舒张压']=np.exp(new['舒张压'])-1
    new['血清甘油三酯']=np.exp(new['血清甘油三酯'])+0.1
    new['血清高密度脂蛋白']=np.exp(new['血清高密度脂蛋白'])
    new['血清低密度脂蛋白']=np.exp2(new['血清低密度脂蛋白'])-1.22
    zz=pd.DataFrame(test_index)
    zz.columns=['vid']
    zz=pd.merge(zz,new,on='vid',how='left')
    #zz.to_csv('../tem/old收缩压lgb_testb.csv')
    #zz.to_csv('../tem/all.csv')
    return zz

def youtrainlgb():
    
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings("ignore")
    
    import time
    import pandas as pd

    
 
    train=pd.read_csv('../data/trainx.csv',encoding='gbk')
    test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',encoding='gbk')
    testt=pd.read_csv('../data/testx.csv',encoding='gbk')
    
    print('haha====================================================')
    # =============================================================================
    # print(train.columns)
    # train=labelencodeall(train)
    # train.to_csv('../tmp/train.csv')
    # =============================================================================
    # =============================================================================
    # train=pd.merge(train,targetall,on='vid',how='left')
    # train=train[train.收缩压.notnull()]
    # testt=train[train.血清高密度脂蛋白.isnull()]
    # =============================================================================
    # =============================================================================
    # data=pd.concat([train,testt])
    # data=zuhe(data)
    # train=data[data.收缩压.notnull()]
    # testt=data[data.收缩压.isnull()]
    # =============================================================================
    print('============================')
    print(train.shape)
    print(test.shape)
    print(testt.shape)
    print('============================')
    train['收缩压'] = train['收缩压'].astype(float)
    train['舒张压'] =train['舒张压'].astype(float)
    train['血清甘油三酯']=train['血清甘油三酯'].astype(float)
    train['收缩压']=np.log(train['收缩压']+1)
    train['舒张压']=np.log(train['舒张压']+1)
    train['血清甘油三酯']=np.log(train['血清甘油三酯']-0.099999999)
    train['血清高密度脂蛋白']=np.log(train['血清高密度脂蛋白'])
    train['血清低密度脂蛋白']=np.log2(train['血清低密度脂蛋白']+1.22001)
    print(train.info())
    train.pop('vid')
    test_index=test.pop('vid')
    testt_index=testt.pop('vid')
    columns=list(train.columns)
    columns.remove('收缩压')
    columns.remove('舒张压')
    columns.remove('血清甘油三酯')
    columns.remove('血清高密度脂蛋白')
    columns.remove('血清低密度脂蛋白')
    print(train.mean())
    
    new=pd.DataFrame()
    new['vid']=testt_index
    col=['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    
    for t in col:
        print(t)
        X=train[columns]
        y=train[t]
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
        gbm = lgb.LGBMRegressor(objective='regression',
                               # num_leaves=23,
                                learning_rate=0.08,
                                #max_depth=25,
                                #max_bin=10000,
                                drop_rate=0.10,
                                
                                #is_unbalance=True,
                                n_estimators=1000)#1000
        gbm.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='l1',
                early_stopping_rounds=100)
        
        y_pred = gbm.predict(testt[columns], num_iteration=gbm.best_iteration)
        print('======================================================================================')
        predictors = [i for i in X_train.columns]
        feat_imp = pd.Series(gbm.feature_importance(), predictors).sort_values(ascending=False)
        print(feat_imp)
        new[t]=y_pred
    
    new['收缩压']=np.exp(new['收缩压'])-1
    new['舒张压']=np.exp(new['舒张压'])-1
    new['血清甘油三酯']=np.exp(new['血清甘油三酯'])+0.1
    new['血清高密度脂蛋白']=np.exp(new['血清高密度脂蛋白'])
    new['血清低密度脂蛋白']=np.exp2(new['血清低密度脂蛋白'])-1.22
    zz=pd.DataFrame(test_index)
    zz.columns=['vid']
    zz=pd.merge(zz,new,on='vid',how='left')
    #zz.to_csv('../tem/old收缩压lgb_testb.csv')
    #zz.to_csv('../tem/all.csv')
    return zz