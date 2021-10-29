# -*- coding: utf-8 -*-
"""
第17章：集成学习
数据获取
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_encode as var_encode
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False  
import warnings
warnings.filterwarnings("ignore") ##忽略警告
##数据读取
def data_read(data_path,file_name):
    df = pd.read_csv( os.path.join(data_path, file_name), delim_whitespace = True, header = None )
    ##变量重命名
    columns = ['status_account','duration','credit_history','purpose', 'amount',
               'svaing_account', 'present_emp', 'income_rate', 'personal_status',
               'other_debtors', 'residence_info', 'property', 'age',
               'inst_plans', 'housing', 'num_credits',
               'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    df.columns = columns
    ##将标签变量由状态1,2转为0,1;0表示好用户，1表示坏用户
    df.target = df.target - 1
      ##数据分为data_train和 data_test两部分，训练集用于得到编码函数，验证集用已知的编码规则对验证集编码
    data_train, data_test = train_test_split(df, test_size=0.2, random_state=0,stratify=df.target)
    return data_train, data_test
##离散变量与连续变量区分   
def category_continue_separation(df,feature_names):
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
    ##先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var,numerical_var
if __name__ == '__main__':
    path = 'D:\\code\\chapter17'
    data_path = os.path.join(path ,'data')
    file_name = 'german.csv'
    ##读取数据
    data_train, data_test = data_read(data_path,file_name)
    sum(data_train.target ==0)
    data_train.target.sum()
    ##区分离散变量与连续变量
    feature_names = list(data_train.columns)
    feature_names.remove('target')
    categorical_var,numerical_var = category_continue_separation(data_train,feature_names)
    
    ###离散变量直接WOE编码
    var_all_bin = list(data_train.columns)
    var_all_bin.remove('target')
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train,data_path,categorical_var, data_train.target,'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(data_test,data_path,categorical_var, data_test.target, 'dict_woe_map',flag='test')
    
    #####连续变量缺失值做填补
    for i in numerical_var:
        if sum(data_train[i].isnull()) >0:
            data_train[i].fillna(data_train[i].mean(),inplace=True)
            

    ###组成分箱后的训练集与测试集
    data_train.reset_index(drop=True,inplace=True)
    data_test.reset_index(drop=True,inplace=True)
    var_1 = numerical_var
    var_1.append('target')
    data_train_1 = pd.concat([df_train_woe[var_woe_name],data_train[var_1]],axis=1)
    data_test_1 = pd.concat([df_test_woe[var_woe_name],data_test[var_1]],axis=1) 
    
    ####取出训练数据与测试数据
    var_all = list(data_train_1.columns)
    var_all.remove('target')

    ####变量归一化
    scaler = StandardScaler().fit(data_train_1[var_all])
    data_train_1[var_all] = scaler.transform(data_train_1[var_all])  
    data_test_1[var_all] = scaler.transform(data_test_1[var_all])

    x_train = np.array(data_train_1[var_all])
    y_train = np.array(data_train_1.target)
    
    x_test = np.array(data_test_1[var_all])
    y_test = np.array(data_test_1.target)
        
   
    ########随机森林模型
    ##设置待优化的超参数
    rf_param = {'n_estimators': list(range(50, 400, 50)),
                'max_depth': list(range(2, 10, 1)),
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    rf_gsearch = GridSearchCV(estimator=RandomForestClassifier(random_state=0, criterion='entropy',
                                                                max_features=0.8, bootstrap=True),
                              param_grid=rf_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    rf_gsearch.fit(x_train, y_train)
    print('RandomForest model best_score_ is {0},and best_params_ is {1}'.format(rf_gsearch.best_score_,
                                                                                 rf_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化随机森林模型
    RF_model_2 = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy',
                                        n_estimators=rf_gsearch.best_params_['n_estimators'],
                                        max_depth=rf_gsearch.best_params_['max_depth'],
                                        max_features=0.8,
                                        min_samples_split=50,
                                        class_weight=rf_gsearch.best_params_['class_weight'],
                                        bootstrap=True)
    ##训练随机森林模型
    RF_model_2_fit = RF_model_2.fit(x_train, y_train)
    
    ##属性
#    ss = RF_model_2_fit.estimators_
#    RF_model_2_fit.classes_
#    RF_model_2_fit.n_features_
#    RF_model_2_fit.feature_importances_
    
    
    ##模型预测
    y_pred = RF_model_2_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
   
    ##变量重要性排序
    ##取出变量重要性结果
    len_1 = len(RF_model_2.feature_importances_)
    var_name = [str(x).split('_woe')[0] for x in var_all]
    ##重要性绘图
    plt.figure(figsize=(10,6))
    fontsize_1 = 14
    plt.barh(np.arange(0,20),RF_model_2.feature_importances_,color =
     'c',tick_label=var_name)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.show()

    
    ###Adaboost模型
    ##设置待优化的超参数
    ada_param = {'n_estimators': list(range(50, 500, 50)),
                'learning_rate': list(np.arange(0.1, 1, 0.2))}
    ##初始化网格搜索
    ada_gsearch = GridSearchCV(estimator=AdaBoostClassifier(algorithm='SAMME.R',random_state=0),
                              param_grid=ada_param, cv=3,  n_jobs=-1, verbose=2)
     ##执行超参数优化
    ada_gsearch.fit(x_train, y_train)
    print('AdaBoostClassifier model best_score_ is {0},and best_params_ is {1}'.format(ada_gsearch.best_score_,
                                                                                 ada_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化Adaboost模型
    ada_model_2 = AdaBoostClassifier( n_estimators=ada_gsearch.best_params_['n_estimators'],
                                        learning_rate=ada_gsearch.best_params_['learning_rate'],
                                        algorithm='SAMME.R',random_state=0)
    ##训练adaboost模型
    ada_model_2_fit = ada_model_2.fit(x_train, y_train)
    
    
    ##模型预测
    y_pred = ada_model_2_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
    ###查看模型训练过程准确率的变化
    ##取出训练过程中模型得分
    n_estimators = ada_gsearch.best_params_['n_estimators']
    per_train = list(ada_model_2_fit.staged_score(x_train, y_train))
    per_test= list(ada_model_2_fit.staged_score(x_test, y_test))
    ##预测准确率绘图
    plt.figure(figsize=(10,6))
    fontsize_1 = 14
    plt.plot(np.arange(0,n_estimators),per_train,'--',color ='c',label='训练集')
    plt.plot(np.arange(0,n_estimators),per_test,':',color ='b',label='测试集')
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.xlabel('n_estimators',fontsize=fontsize_1)
    plt.ylabel('score',fontsize=fontsize_1)
    plt.legend(fontsize=fontsize_1)
    plt.show()
    
    ####GBDT模型
    ##设置待优化的超参数
    gbdt_param = {'n_estimators': list(range(50, 400, 50)),
                'max_depth': list(range(2, 5, 1)),
                'learning_rate': list(np.arange(0.01, 0.5, 0.05)) }
    ##初始化网格搜索
    gbdt_gsearch = GridSearchCV(estimator=GradientBoostingClassifier( subsample=0.8,max_features=0.8, validation_fraction=0.1, 
                                                                   n_iter_no_change =3,random_state=0),param_grid=gbdt_param, 
                                                                   cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    gbdt_gsearch.fit(x_train, y_train)
    print('gbdt model best_score_ is {0},and best_params_ is {1}'.format(gbdt_gsearch.best_score_,
                                                                                 gbdt_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化GBDT模型
    GBDT_model= GradientBoostingClassifier(subsample=0.8,max_features=0.8, validation_fraction=0.1, 
                                                      n_iter_no_change =3,random_state=0 ,
                                        n_estimators=gbdt_gsearch.best_params_['n_estimators'],
                                        max_depth=gbdt_gsearch.best_params_['max_depth'],
                                        learning_rate=gbdt_gsearch.best_params_['learning_rate'])
    ##训练GBDT模型
    GBDT_model_fit = GBDT_model.fit(x_train, y_train)
    
    
    ###看一下混沌矩阵
    y_pred = GBDT_model_fit.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
    ###xgboost模型
    ##设置待优化的超参数
    xgb_param = {'max_depth': list(range(2, 6, 1)), 'min_child_weight': list(range(1, 4, 1)),
                 'learning_rate': list(np.arange(0.01, 0.3, 0.05)), 'scale_pos_weight': list(range(1, 5, 1))}
    ##初始化网格搜索
    xgb_gsearch = GridSearchCV(
        estimator=XGBClassifier(random_state=0, n_estimators=500, subsample=0.8, colsample_bytree=0.8),
        param_grid=xgb_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    xgb_gsearch.fit(x_train, y_train)
    print('xgboost model best_score_ is {0},and best_params_ is {1}'.format(xgb_gsearch.best_score_,
                                                                          xgb_gsearch.best_params_))
    ##用最优参数，初始化xgboost模型
    xgboost_model = XGBClassifier(random_state=0, n_jobs=-1,
                                    n_estimators=500,
                                    max_depth=xgb_gsearch.best_params_['max_depth'],
                                    subsample=0.8, colsample_bytree=0.8,
                                    learning_rate=xgb_gsearch.best_params_['learning_rate'],
                                    scale_pos_weight=xgb_gsearch.best_params_['scale_pos_weight'])
    ##训练xgboost模型
    xgboost_model_fit = xgboost_model.fit(x_train, y_train)

    
    ##模型预测
    y_pred = xgboost_model_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
    ##给出概率预测结果
    y_score_test = xgboost_model_fit.predict_proba(x_test)[:, 1]
    ##计算AR。gini等
    fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    ar = 2*roc_auc-1
    print('test set:  model AR is {0},and ks is {1},auc={2}'.format(ar,
                 ks,roc_auc)) 
    ####ks曲线
    plt.figure(figsize=(10,6))
    fontsize_1 = 12
    plt.plot(np.linspace(0,1,len(tpr)),tpr,'--',color='black')
    plt.plot(np.linspace(0,1,len(tpr)),fpr,':',color='black')
    plt.plot(np.linspace(0,1,len(tpr)),tpr - fpr,'-',color='grey')
    plt.grid()
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.xlabel('概率分组',fontsize=fontsize_1)
    plt.ylabel('累积占比%',fontsize=fontsize_1)
    
    
    
    
    
    
