# -*- coding: utf-8 -*-
"""
第18章：模型融合stacking
数据获取
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_encode as var_encode
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
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
    path = 'D:\\code\\chapter18'
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
        
    ########stacking模型融合
    kfold = KFold(n_splits=5)
    ##存放概率预测结果
    y_pred_1 = np.zeros(x_train.shape[0])
    y_pred_2 = np.zeros(x_train.shape[0])
    y_pred_3 = np.zeros(x_train.shape[0])
    for train_index, test_index in kfold.split(x_train):
        ##K折中的建模部分
        x_train_temp = x_train[train_index,:]
        y_train_temp = y_train[train_index]
        ##K折中的预测部分
        x_pre_temp = x_train[test_index,:]
        y_pre_temp = y_train[test_index]
        
        ##一级模型
        ##随机森林模型
        RF_model_1 = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy',
                                            n_estimators=100,
                                            max_depth=3,
                                            max_features=0.8,
                                            min_samples_split=50,
                                            class_weight={1: 2, 0: 1},
                                            bootstrap=True)
        RF_model_fit_1 = RF_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_1[test_index] = RF_model_fit_1.predict_proba(x_pre_temp)[:, 1]
        
        ##xgboost模型
        xgboost_model_1 = XGBClassifier(random_state=0, n_jobs=-1,
                                        n_estimators=100,
                                        max_depth=2,
                                        min_child_weight=2,
                                        subsample=0.8, colsample_bytree=0.8,
                                        learning_rate=0.02,
                                        scale_pos_weight=2)
        xgboost_model_fit_1 = xgboost_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_2[test_index] = xgboost_model_fit_1.predict_proba(x_pre_temp)[:, 1]
         
        ##svm模型
        svm_model_1 = SVC(kernel='rbf',C =0.5,gamma=0.1,class_weight= {1: 2, 0: 1},probability=True)
        svm_model_fit_1 = svm_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_3[test_index] = svm_model_fit_1.predict_proba(x_pre_temp)[:, 1]
        
    ##将预测概率输出与原特征合并，构造新的训练集
    y_pred_all = np.vstack([y_pred_1, y_pred_2,y_pred_3]).T
    x_train_1 = np.hstack([x_train,y_pred_all])
    
    ##训练二级模型
    ##设置待优化的超参数
    lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lr_gsearch.fit(x_train_1, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    
    ##最有参数训练模型
    ##用最优参数，初始化Logistic回归模型
    LR_model_2 = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])
    ##训练Logistic回归模型
    LR_model_fit = LR_model_2.fit(x_train_1, y_train)
    
    
    ##测试集预测结果
    ##先用一级模型进行概率预测
    ##给出概率预测
    ytest_pred_1= RF_model_fit_1.predict_proba(x_test)[:, 1]
    ytest_pred_2 = xgboost_model_fit_1.predict_proba(x_test)[:, 1]
    ytest_pred_3 = svm_model_fit_1.predict_proba(x_test)[:, 1]
    ##构造新特征矩阵
    ytest_pred_all = np.vstack([ytest_pred_1, ytest_pred_2,ytest_pred_3]).T
    x_test_1 = np.hstack([x_test,ytest_pred_all])
    
    ##用二级模型进行预测
    y_pred = LR_model_fit.predict(x_test_1)
    y_pred_proba = LR_model_fit.predict_proba(x_test_1)[:, 1]
      
    ###看一下混沌矩阵
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value))
    
    ##计算Ks值、AR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    ar = 2*roc_auc-1
    print('test set:  model AR is {0},and ks is {1},auc={2}'.format(ar,
                 ks,roc_auc)) 
    ##绘制K-s曲线
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

    
    
    
 
    
    
    
    
    
    
    
