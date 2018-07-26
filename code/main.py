# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:59:25 2018

@author: cjzhao_13
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing
import time
import pandas as pd
import datetime
import getfeature
import trainlgb
import trainxgb

print('特征构造')
preprocess()
print('preprocess 完成')
againpreprocess()
print('againpreprocess 完成')
print('模型训练')
lgb_result=mytrainlgb()
xgb_result=mytrainxgb()

lgb_result['收缩压']=0.48*lgb_result['收缩压']+0.52*xgb_result['收缩压']
lgb_result['舒张压']=0.48*lgb_result['舒张压']+0.52*xgb_result['舒张压']
lgb_result['血清甘油三酯']=0.48*lgb_result['血清甘油三酯']+0.52*xgb_result['血清甘油三酯']
lgb_result['血清高密度脂蛋白']=0.48*lgb_result['血清高密度脂蛋白']+0.52*xgb_result['血清高密度脂蛋白']
lgb_result['血清低密度脂蛋白']=0.48*lgb_result['血清低密度脂蛋白']+0.52*xgb_result['血清低密度脂蛋白']

lgb_result2=youtrainlgb()
xgb_result2=youtrainxgb()

lgb_result2['收缩压']=0.48*lgb_result2['收缩压']+0.52*xgb_result2['收缩压']
lgb_result2['舒张压']=0.48*lgb_result2['舒张压']+0.52*xgb_result2['舒张压']
lgb_result2['血清甘油三酯']=0.48*lgb_result2['血清甘油三酯']+0.52*xgb_result2['血清甘油三酯']
lgb_result2['血清高密度脂蛋白']=0.48*lgb_result2['血清高密度脂蛋白']+0.52*xgb_result2['血清高密度脂蛋白']
lgb_result2['血清低密度脂蛋白']=0.48*lgb_result2['血清低密度脂蛋白']+0.52*xgb_result2['血清低密度脂蛋白']

lgb_result['收缩压']=0.45*lgb_result['收缩压']+0.55*lgb_result2['收缩压']
lgb_result['舒张压']=0.45*lgb_result['舒张压']+0.55*lgb_result2['舒张压']
lgb_result['血清甘油三酯']=0.45*lgb_result['血清甘油三酯']+0.55*lgb_result2['血清甘油三酯']
lgb_result['血清高密度脂蛋白']=0.45*lgb_result['血清高密度脂蛋白']+0.55*lgb_result2['血清高密度脂蛋白']
lgb_result['血清低密度脂蛋白']=0.45*lgb_result['血清低密度脂蛋白']+0.55*lgb_result2['血清低密度脂蛋白']


ISOTIMEFORMAT='%Y%m%d'
ISOTIMEFORMAT1='%H%M%S'
str1='../submit/submit_'+(time.strftime(ISOTIMEFORMAT))+'_'+(time.strftime(ISOTIMEFORMAT1))+'.csv'
lgb_result.to_csv(str1,index=None,header=None)