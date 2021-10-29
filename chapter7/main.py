# -*- coding: utf-8 -*-
"""
第7章：变量选择
数据获取
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_bin_methods as varbin_meth
import variable_encode as var_encode
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use(arg='Qt5Agg')
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from feature_selector import FeatureSelector
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
    path = 'D:\\code\\chapter7'
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
    for s in set(numerical_var):
        print('变量'+s+'可能取值'+str(len(data_train[s].unique())))
        if len(data_train[s].unique())<=10:
            categorical_var.append(s)
            numerical_var.remove(s)
            ##同时将后加的数值变量转为字符串
            index_1 = data_train[s].isnull()
            if sum(index_1) > 0:
                data_train.loc[~index_1,s] = data_train.loc[~index_1,s].astype('str')
            else:
                data_train[s] = data_train[s].astype('str')
            index_2 = data_test[s].isnull()
            if sum(index_2) > 0:
                data_test.loc[~index_2,s] = data_test.loc[~index_2,s].astype('str')
            else:
                data_test[s] = data_test[s].astype('str')

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
    df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,data_path,
                                 var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
    
    
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,data_path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')
    y = np.array(data_train_bin.target)
    
    ###过滤法特征选择
    ##方差筛选
    df_train_woe = df_train_woe[var_woe_name]
    len_1 = df_train_woe.shape[1]
    select_var = VarianceThreshold(threshold=0.01)
    select_var_model = select_var.fit(df_train_woe)
    df_1 = pd.DataFrame(select_var_model.transform(df_train_woe))
    ##保留的索引
    save_index = select_var.get_support(True)
    var_columns = [list(df_train_woe.columns)[x] for x in save_index]
    df_1.columns = var_columns
    ##删除变量的方差
    select_var.variances_[[x for x in range(len_1) if x not in save_index]]
    [list(df_train_woe.columns)[x] for x in range(len_1) if x not in save_index]
    [select_var.variances_[x] for x in range(len_1) if x not in save_index]

    ##单变量筛选
    select_uinvar = SelectKBest(score_func= f_classif, k=15)
    select_uinvar_model = select_uinvar.fit(df_train_woe,y)
    df_1 = select_uinvar_model.transform(df_train_woe)
    ##看得分
    len_1 = len(select_uinvar_model.scores_)
    var_name = [str(x).split('_BIN_woe')[0] for x in list(df_train_woe.columns)]
    ##
    plt.figure(figsize=(10,6))
    fontsize_1 = 14
    plt.barh(np.arange(0,len_1),select_uinvar_model.scores_,color = 'c',tick_label=var_name)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.xlabel('得分',fontsize=fontsize_1)
    plt.show()
    
    ##分析变量相关性
    ##计算相关矩阵
    correlations = abs(df_train_woe.corr())
    ##相关性绘图
    fig = plt.figure(figsize=(10,6)) 
    fontsize_1 = 10
    sns.heatmap(correlations,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
    plt.xticks(np.arange(20)+0.5,var_name,fontsize=fontsize_1,rotation=20)
    plt.yticks(np.arange(20)+0.5,var_name,fontsize=fontsize_1) 
    plt.title('相关性分析')
#    plt.xlabel('得分',fontsize=fontsize_1)
    plt.show()

    ##包装法变量选择：递归消除法 
    ##给定学习器
    estimator = SVR(kernel="linear")
    ##递归消除法 
    select_rfecv = RFECV(estimator, step=1, cv=3)
    select_rfecv_model = select_rfecv.fit(df_train_woe, y)
    df_1 = pd.DataFrame(select_rfecv_model.transform(df_train_woe))
    ##查看结果
    select_rfecv_model.support_ 
    select_rfecv_model.n_features_
    select_rfecv_model.ranking_
    
    ###嵌入法变量选择
    ##选择学习器
    lr = LogisticRegression(C=0.1, penalty='l2')
    ##嵌入法变量选择
    select_lr = SelectFromModel(lr, prefit=False,threshold ='mean')
    select_lr_model = select_lr.fit(df_train_woe, y)
    df_1 = pd.DataFrame(select_lr_model.transform(df_train_woe))
    ##查看结果
    select_lr_model.threshold_   
    select_lr_model.get_support(True)
    
    ##基学习器选择预训练的决策树来进行变量选择
    ##先训练决策树
    cart_model = DecisionTreeClassifier(criterion='gini',max_depth = 3).fit(df_train_woe, y)
    cart_model.feature_importances_  
    ##用预训练模型进行变量选择
    select_dt_model = SelectFromModel(cart_model, prefit=True)
    df_1 = pd.DataFrame(select_dt_model.transform(df_train_woe))
    ##查看结果
    select_dt_model.get_support(True)
    
    ###集成学习下的变量选择lightgbm     
    fs = FeatureSelector(data = df_train_woe, labels = y)
    ##设置筛选参数
    fs.identify_all(selection_params = {'missing_threshold': 0.9, 
                                         'correlation_threshold': 0.6, 
                                         'task': 'classification', 
                                         'eval_metric': 'binary_error',
                                         'max_depth':2,
                                         'cumulative_importance': 0.90})
    df_train_woe = fs.remove(methods = 'all')
    ##查看结果
    fs.feature_importances
    fs.corr_matrix
    fs.record_low_importance
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    
    
    
    
    
    
    
    
