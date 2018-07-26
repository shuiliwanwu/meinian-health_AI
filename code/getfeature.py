# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:47:27 2018

@author: cjzhao_13
"""

import pandas as pd 
import numpy as np 

def preprocess():
        
    def read_data(path1, path2):
        data_dict = {}
        data_file1 = open(path1, "r", encoding="utf-8")
        data_text1 = data_file1.read()[1:].split("\n")[1:-1]
        data_file2 = open(path2, "r", encoding="utf-8")
        data_text2 = data_file2.read()[1:].split("\n")[1:-1]
        data_text1.extend(data_text2)
        for line in data_text1:
            line = line.split("$")
            if line[0] in data_dict:
                data_dict[line[0]][line[1]] = line[2]
            else:
                data_dict[line[0]] = {}
        return pd.DataFrame(data_dict).T
    
    def labelencodeall(traintest):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        aa=traintest.select_dtypes(include=['object'], exclude=None)
        columns=list(aa.columns)
        for col in columns:
            if (col is not 'vid') and (col is not 'is_trade' ):
                traintest[col]=[str(t) for t in traintest[col]]
                traintest[col]=le.fit_transform(traintest[col])
        return traintest  
    
    
    path1='../data/meinian_round1_data_part1_20180408.txt'
    path2='../data/meinian_round1_data_part2_20180408.txt'
    aa=read_data(path1, path2)
    data=aa.copy()
    data=data.reset_index()
    data['vid']=data['index']
    data.pop('index')
    data=labelencodeall(data)
    
    target=pd.read_csv('../data/meinian_round1_train_20180408.csv',encoding='gbk')
    dd=target.copy()
    dd=dd[(dd['收缩压']!='未查')&(dd['收缩压']!='弃查')&(dd['收缩压']!='0')]
    dd=dd[(dd['舒张压']!='未查')&(dd['舒张压']!='弃查')&(dd['舒张压']!='974')&(dd['舒张压']!='0')&(dd['舒张压']!='100164')]
    col=['> 11.00','> 6.22','12.51++','16.04++','17.60++','2.2.8','2.23+','2.40+','2.45+','2.46+',
        '2.48+','2.63+','2.78+','3.17+','3.25+','3.36+','3.61+','3.64+','3.81+','3.96+','4.09+','4.13+',
         '4.15+','4.19+','4.22+','4.34+','4.46+','4.57+','4.64+','4.75+','5.02+','5.53+','5.77+','6.15++',
         '6.47++','7.15+','7.75轻度乳糜','8.28+','8.62++','9.14++','0.1']
    for t in col:
        dd=dd[(dd['血清甘油三酯']!=t)]
        
    dd=dd[(dd['血清低密度脂蛋白']>0)]
    dd['收缩压']=dd['收缩压'].astype('float')
    dd['舒张压']=dd['舒张压'].astype('float')
    dd['血清甘油三酯']=dd['血清甘油三酯'].astype('float')
    target=dd.copy()
    traintest=pd.merge(data,target,on='vid',how='left')
    train=traintest[traintest.收缩压.notnull()]
    test=traintest[traintest.收缩压.isnull()]
    print(train.shape,test.shape)
    train.to_csv('../data/train.csv',index=None)
    test.to_csv('../data/test.csv',index=None)

def againpreprocess():
    
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
    import datetime
    import time
    
    def zuhe(data):
        
        columns=['313','3190','1840','193','312','31','32','33','183','2405','1115','193','192','191','10004','190','1815','1850','1814','0102','2404','1117','2403','10002','314','319']    
        for col in columns:
            data[col]=data[col].astype('float64')
        print('begin')
        for cloi in columns:
            for cloj in columns:
                if cloi is not cloj:
                    data[cloi+'+'+cloj]=data[cloi]+data[cloj]#le.fit_transform(data[cloi]+data[cloj])
                    data[cloi+'-'+cloj]=data[cloi]-data[cloj]
                    data[cloi+'*'+cloj]=data[cloi]*data[cloj]
                    data[cloi+'/'+cloj]=data[cloi]/data[cloj]
        return data
    
    
        
    # 读取数据
    part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep='$')
    part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep='$')
    part_1_2 = pd.concat([part_1,part_2])
    part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)

    print('begin')
    # 重复数据的拼接操作
    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df
    # 数据简单处理
    print('find_is_copy')
    print(part_1_2.shape)
    is_happen = part_1_2.groupby(['vid','table_id']).size().reset_index()
    # 重塑index用来去重
    is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
    is_happen_new = is_happen[is_happen[0]>1]['new_index']
    
    part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']
    
    unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
    unique_part = unique_part.sort_values(['vid','table_id'])
    no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
    print('begin')
    part_1_2_not_unique = unique_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
    part_1_2_not_unique.rename(columns={0:'field_results'},inplace=True)
    print('xxx')
    tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','table_id','field_results']]])
    # 行列转换
    print('finish')
    tmp = tmp.pivot(index='vid',values='field_results',columns='table_id')
    tmp.to_csv('../data/tmp.csv')
    
    df_train=pd.read_csv('../data/tmp.csv',low_memory=False,encoding='gbk')
    abnormal=pd.read_csv('abnormal.csv')
    normal= pd.read_csv('normal.csv')
    missing_df = df_train.isnull().sum(axis=0).reset_index()
    
    # missing <37000
    
    tmp=missing_df[missing_df[0]<37000]
    
    print(tmp.shape)
    
    
    
    # get new train data with 144 cols
    
    
    train_feat=df_train[tmp['index'].values]
    
    abnor = abnormal['res']
    
    nor = normal['res']
    # abnormal key words replace 2
    
    
    def abnormalReplace(x):
        for c in abnor:
            if(str(x).find(c)>=0):
                return 2
            
        for d in nor:
            if(str(x).find(d)>=0):
                return 1
        if(is_number(x)==False):
            return 0
        else:
            return x
    
    
        
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
     
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
     
        return False
    
    for cols in train_feat.columns:
        if cols not in ['vid']:
            train_feat[cols] =train_feat[cols].map(abnormalReplace)
    print(train_feat.head())
    data=train_feat.copy()
    
    data=zuhe(data)
    
    target=pd.read_csv('../data/meinian_round1_train_20180408.csv',encoding='gbk')
    
    dd=target.copy()
    
    dd=dd[(dd['收缩压']!='未查')&(dd['收缩压']!='弃查')&(dd['收缩压']!='0')]
    dd=dd[(dd['舒张压']!='未查')&(dd['舒张压']!='弃查')&(dd['舒张压']!='974')&(dd['舒张压']!='0')&(dd['舒张压']!='100164')]
    col=['> 11.00','> 6.22','12.51++','16.04++','17.60++','2.2.8','2.23+','2.40+','2.45+','2.46+',
        '2.48+','2.63+','2.78+','3.17+','3.25+','3.36+','3.61+','3.64+','3.81+','3.96+','4.09+','4.13+',
         '4.15+','4.19+','4.22+','4.34+','4.46+','4.57+','4.64+','4.75+','5.02+','5.53+','5.77+','6.15++',
         '6.47++','7.15+','7.75轻度乳糜','8.28+','8.62++','9.14++','0.1']
    for t in col:
        dd=dd[(dd['血清甘油三酯']!=t)]
        
    dd=dd[(dd['血清低密度脂蛋白']>0)]
    dd['收缩压']=dd['收缩压'].astype('float')
    dd['舒张压']=dd['舒张压'].astype('float')
    dd['血清甘油三酯']=dd['血清甘油三酯'].astype('float')
    target=dd.copy()
    data=pd.merge(data,target,on='vid',how='left')
    train=data[data.收缩压.notnull()]
    testt=data[data.收缩压.isnull()]
    print(train.shape)
    print(testt.shape)
    train.pop('收缩压')
    train.pop('舒张压')
    train.pop('血清甘油三酯')
    train.pop('血清高密度脂蛋白')
    train.pop('血清低密度脂蛋白')
    testt.pop('收缩压')
    testt.pop('舒张压')
    testt.pop('血清甘油三酯')
    testt.pop('血清高密度脂蛋白')
    testt.pop('血清低密度脂蛋白')
    
    columns=list(train.columns)
    
    for i in range(1,len(columns)):
        if columns[i] is not 'vid':
            columns[i]='new'+columns[i]
            
    train.columns=columns
    testt.columns=columns
    
    train1=pd.read_csv('../data/train.csv',encoding='gbk')
    testt1=pd.read_csv('../data/test.csv',encoding='gbk')
    trainx=pd.merge(train1,train,on='vid',how='left')
    testx=pd.merge(testt1,testt,on='vid',how='left')
    trainx.to_csv('../data/trainx.csv',index=None)
    testx.to_csv('../data/testx.csv',index=None)