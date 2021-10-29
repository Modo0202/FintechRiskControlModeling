# -*- coding: utf-8 -*-
"""
第17章：SVM模型
数据获取
"""
import os
import sys
#path = __file__
#path = os.path.abspath(path + ((os.sep + '..') * 2))
#sys.path.append(path)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_encode as var_encode
import numbers
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
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
    data_train, data_test = train_test_split(df, test_size=0.2, random_state=100,stratify=df.target)
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
        if sum(data_test[i].isnull()) >0:
            data_test[i].fillna(data_test[i].mean(),inplace=True)

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
        
   
    ########线性支持向量机模型
    ##设置待优化的超参数
    lin_svm_param = {'C': np.arange(0.1,5,0.1),
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]  }
    ##初始化网格搜索
    lin_svm_gsearch = GridSearchCV(estimator=LinearSVC(),param_grid=lin_svm_param, 
                  cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lin_svm_gsearch.fit(x_train, y_train)
    print('linearSVC model best_score_ is {0},and best_params_ is {1}'.format(lin_svm_gsearch.best_score_,
                                                                             lin_svm_gsearch.best_params_))
    
    ##用最优参数，初始化模型
    lin_svm_model = LinearSVC(C = lin_svm_gsearch.best_params_['C'],
                                    class_weight=lin_svm_gsearch.best_params_['class_weight'])
    ##模型训练
    lin_svm_model_fit = lin_svm_model.fit(x_train, y_train)
    
    
    ##模型预测
    y_pred = lin_svm_model_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
   
    #########非线性支持向量机模型
    ##设置待优化的超参数
    svm_param = {'C': np.arange(0.1,5,0.1),
                 'gamma':np.arange(0.01,10,1),
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]  }
    ##初始化网格搜索
    svm_gsearch = GridSearchCV(estimator=SVC(kernel='rbf'),param_grid=svm_param, 
                  cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    ##执行超参数优化
    svm_gsearch.fit(x_train, y_train)
    print('DecisionTreeClassifier model best_score_ is {0},and best_params_ is {1}'.format(svm_gsearch.best_score_,
                                                                             svm_gsearch.best_params_))
    
    ##用最优参数，初始化模型
    svm_model = SVC(kernel='rbf',C =svm_gsearch.best_params_['C'],gamma=svm_gsearch.best_params_['gamma'],
                                    class_weight=svm_gsearch.best_params_['class_weight'],probability=True)
     ##模型训练
    svm_model_fit = svm_model.fit(x_train, y_train)
    
     ##模型预测
    y_pred = svm_model_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value))
    
    ##给出概率预测结果
    y_score_test = svm_model_fit.predict_proba(x_test)[:, 1]
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

    
    
    
 
    
    
    
    
    
    
    
