# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:25:42 2018

@author: cjzhao_13
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 22:10:35 2018

@author: cjzhao_13
"""
def mytrainxgb():
    
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings("ignore")
    
    import time
    import pandas as pd
    import datetime
    import time
    
    
    
    
    def labelencodeall(traintest):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        aa=traintest.select_dtypes(include=['object'], exclude=None)
        columns=list(aa.columns)
        for col in columns:
            if (col is not 'vid') and (col is not 'is_trade' )and(col is not 'context_timestamp_date')and(col is not 'context_timestamp_hour'):
                traintest[col]=[str(t) for t in traintest[col]]
                traintest[col]=le.fit_transform(traintest[col])
        return traintest  
    
    def zuhe(data):
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        columns=['2405','193','192','191','10004','190','1815','1850','1814','102','2404','1106','1107','1117','101','2403','669021','424','10002','314','319']    
        for col in columns:
            data[col] = data[col].astype(str)
        
    
        print('begin')
        for cloi in columns:
            for cloj in columns:
                if cloi is not cloj:
                    data[cloi+cloj]=le.fit_transform(data[cloi]+data[cloj])
                
                
     
        for col in columns:
            data[col] = data[col].astype(float)
    
        return data
    

    train=pd.read_csv('../data/train.csv',encoding='gbk')
    test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',encoding='gbk')
    testt=pd.read_csv('../data/test.csv',encoding='gbk')

    
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
    
    new=pd.DataFrame()
    new['vid']=testt_index
    col=['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    for t in col:
        
        X=train[columns]
        y=train[t]
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
        gbm = xgb.XGBRegressor(learning_rate=0.08,gamma=0.2, # 0.02£¬1000    0.1£¬150
                               max_depth=5, subsample=0.9, # 7£¬8
                               colsample_bytree=0.8,
                               min_child_weight=7, scale_pos_weight=2,
                               n_estimators=1279,#1279
                               reg_alpha=2, reg_lambda=0.5,
                               nthread = -1
                                    )
        gbm.fit(X_train, y_train, eval_metric='mae', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=50)  
       
        y_pred = gbm.predict(testt[columns], gbm.best_iteration)
        new[t]=y_pred
    
    
    
    new['收缩压']=np.exp(new['收缩压'])-1
    new['舒张压']=np.exp(new['舒张压'])-1
    new['血清甘油三酯']=np.exp(new['血清甘油三酯'])+0.1
    new['血清高密度脂蛋白']=np.exp(new['血清高密度脂蛋白'])
    new['血清低密度脂蛋白']=np.exp2(new['血清低密度脂蛋白'])-1.22
    zz=pd.DataFrame(test_index)
    zz.columns=['vid']
    zz=pd.merge(zz,new,on='vid',how='left')
    
    return zz


def youtrainxgb():
    
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings("ignore")
    
    import time
    import pandas as pd
    import datetime
    import time
    
    
    
    train=pd.read_csv('../data/trainx.csv',encoding='gbk')
    test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',encoding='gbk')
    testt=pd.read_csv('../data/testx.csv',encoding='gbk')

    
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
    
    new=pd.DataFrame()
    new['vid']=testt_index
    col=['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    for t in col:
        
        X=train[columns]
        y=train[t]
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)
        gbm = xgb.XGBRegressor(learning_rate=0.08,gamma=0.2, # 0.02£¬1000    0.1£¬150
                               max_depth=5, subsample=0.9, # 7£¬8
                               colsample_bytree=0.8,
                               min_child_weight=7, scale_pos_weight=2,
                               n_estimators=1279,#1279
                               reg_alpha=2, reg_lambda=0.5,
                               nthread = -1
                                    )
        gbm.fit(X_train, y_train, eval_metric='mae', verbose = True, eval_set = [(X_test, y_test)],early_stopping_rounds=50)  
       
        y_pred = gbm.predict(testt[columns], gbm.best_iteration)
        new[t]=y_pred
    
    
    
    new['收缩压']=np.exp(new['收缩压'])-1
    new['舒张压']=np.exp(new['舒张压'])-1
    new['血清甘油三酯']=np.exp(new['血清甘油三酯'])+0.1
    new['血清高密度脂蛋白']=np.exp(new['血清高密度脂蛋白'])
    new['血清低密度脂蛋白']=np.exp2(new['血清低密度脂蛋白'])-1.22
    zz=pd.DataFrame(test_index)
    zz.columns=['vid']
    zz=pd.merge(zz,new,on='vid',how='left')
    
    return zz