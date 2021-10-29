# -*- coding: utf-8 -*-
"""
第14章：特征工程  算法层特征工程：添加树特征
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_bin_methods as varbin_meth
import variable_encode as var_encode
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
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
def func_s(x):
    return str(x[0])+'_Cross_'+str(x[1])
if __name__ == '__main__':
    path = 'D:\\code\\chapter13'
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
    
    ###连续变量分箱
    dict_cont_bin = {}
    for i in numerical_var:
        print(i)
        dict_cont_bin[i],gain_value_save , gain_rate_save = varbin_meth.cont_var_bin(data_train[i], data_train.target, method=2, mmin=3, mmax=12,
                                     bin_rate=0.01, stop_limit=0.05, bin_min_num=20)
    ###离散变量分箱
    dict_disc_bin = {}
    del_key = []
    for i in categorical_var:
        dict_disc_bin[i],gain_value_save , gain_rate_save ,del_key_1 = varbin_meth.disc_var_bin(data_train[i], data_train.target, method=2, mmin=3,
                                     mmax=8, stop_limit=0.05, bin_min_num=20)
        if len(del_key_1)>0 :
            del_key.extend(del_key_1)
    ###删除分箱数只有1个的变量
    if len(del_key) > 0:
        for j in del_key:
            del dict_disc_bin[j]
    
    ##训练数据分箱
    ##连续变量分箱映射
    df_cont_bin_train = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_train = pd.concat([ df_cont_bin_train , varbin_meth.cont_var_bin_map(data_train[i], dict_cont_bin[i]) ], axis = 1)
    ##离散变量分箱映射
#    ss = data_train[list( dict_disc_bin.keys())]
    df_disc_bin_train = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_train = pd.concat([ df_disc_bin_train , varbin_meth.disc_var_bin_map(data_train[i], dict_disc_bin[i]) ], axis = 1)

  
    ##测试数据分箱
    ##连续变量分箱映射
    df_cont_bin_test = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_test = pd.concat([ df_cont_bin_test , varbin_meth.cont_var_bin_map(data_test[i], dict_cont_bin[i]) ], axis = 1)
    ##离散变量分箱映射
#    ss = data_test[list( dict_disc_bin.keys())]
    df_disc_bin_test = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_test = pd.concat([ df_disc_bin_test , varbin_meth.disc_var_bin_map(data_test[i], dict_disc_bin[i]) ], axis = 1)
    
    ###组成分箱后的训练集与测试集
    df_disc_bin_train['target'] = data_train.target
    data_train_bin = pd.concat([df_cont_bin_train,df_disc_bin_train],axis=1)
    df_disc_bin_test['target'] = data_test.target
    data_test_bin = pd.concat([df_cont_bin_test,df_disc_bin_test],axis=1)

    data_train_bin.reset_index(inplace=True,drop=True)
    data_test_bin.reset_index(inplace=True,drop=True)

    ###WOE编码
    var_all_bin = list(data_train_bin.columns)
    var_all_bin.remove('target')
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,data_path,var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,data_path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')

    ####取出训练数据与测试数据
    x_train = df_train_woe[var_woe_name]
    x_train = np.array(x_train)
    y_train = np.array(data_train_bin.target)
    
    x_test = df_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(data_test_bin.target)
        
    ####GBDT模型
    GBDT_model= GradientBoostingClassifier(subsample=0.8,max_features=0.8, validation_fraction=0.1, 
                                    n_iter_no_change =3,random_state=0,n_estimators=20,
                                     max_depth=2,learning_rate=0.1)
    ##训练GBDT模型
    GBDT_model_fit = GBDT_model.fit(x_train, y_train)
    
    ###用apply方法得到树的映射结果
    train_new_feature= GBDT_model_fit.apply(x_train)[:, :, 0]
    test_new_feature= GBDT_model_fit.apply(x_test)[:, :, 0]
    np.unique(train_new_feature[:,1])
    ##进行One-hot编码
    enc = OneHotEncoder(dtype='int').fit(train_new_feature)
    df_train = pd.DataFrame( enc.transform(train_new_feature).toarray())
    df_test = pd.DataFrame( enc.transform(test_new_feature).toarray())
    ##合并得到新的数据集
    x_train_1 = np.hstack([x_train,df_train])
    x_test_1 = np.hstack([x_test,df_test])
    
    
    ########logistic模型
    ##参数优化
    lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    lr_gsearch.fit(x_train_1, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    
    ##最优参数训练模型
    LR_model_2 = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])
    
    LR_model_fit = LR_model_2.fit(x_train_1, y_train)
    
  
    ###看一下混沌矩阵
    y_pred = LR_model_fit.predict(x_test_1)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    ##计算fpr与tpr
    y_score_test = LR_model_fit.predict_proba(x_test_1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
    ####计算AR。gini等
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    ar = 2*roc_auc-1
    gini = ar
    print('test set:  model AR is {0},and ks is {1}'.format(ar,
                 ks)) 
    ####ks曲线
    plt.figure(figsize=(10,6))
    fontsize_1 = 12
    plt.plot(np.linspace(0,1,len(tpr)),tpr,'--',color='black', label='正样本洛伦兹曲线')
    plt.plot(np.linspace(0,1,len(tpr)),fpr,':',color='black', label='负样本洛伦兹曲线')
    plt.plot(np.linspace(0,1,len(tpr)),tpr - fpr,'-',color='grey')
    plt.grid()
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.xlabel('概率分组',fontsize=fontsize_1)
    plt.ylabel('累积占比%',fontsize=fontsize_1)
    plt.legend(fontsize=fontsize_1)
    print( max(tpr - fpr))
        
    
    

