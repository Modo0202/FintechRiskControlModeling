# -*- coding: utf-8 -*-
"""
第14章：FM特征交叉
数据获取
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import variable_encode as var_encode
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
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
    
#    ###离散变量直接WOE编码
#    var_all_bin = list(data_train.columns)
#    var_all_bin.remove('target')
#    ##训练集WOE编码
#    df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train,data_path,categorical_var, data_train.target,'dict_woe_map', flag='train')
#    ##测试集WOE编码
#    df_test_woe, var_woe_name = var_encode.woe_encode(data_test,data_path,categorical_var, data_test.target, 'dict_woe_map',flag='test')
#    
    #####连续变量缺失值做填补
    for i in numerical_var:
        if sum(data_train[i].isnull()) >0:
            data_train[i].fillna(data_train[i].mean(),inplace=True)
            
    ####变量归一化
    scaler = StandardScaler().fit(data_train[numerical_var])
    data_train[numerical_var] = scaler.transform(data_train[numerical_var])  
    data_test[numerical_var] = scaler.transform(data_test[numerical_var])
    
#    data_train = data_train_1
#    data_test = data_test_1
    
    ####取出训练数据与测试数据
    var_all = list(data_train.columns)
    var_all.remove('target')
    df_all = pd.concat([data_train,data_test],axis=0)
    ###df转为字典
    df_all =  df_all[var_all].to_dict(orient='records')
    x_train =  data_train[var_all].to_dict(orient='records')
    x_test =  data_test[var_all].to_dict(orient='records')
    ##字典转为稀疏矩阵
    model_dictV = DictVectorizer().fit(df_all)
    x_train = model_dictV.fit_transform(x_train)
    x_test = model_dictV.transform(x_test)
    
    y_train = np.array(data_train.target)
    y_test = np.array(data_test.target)
    x_train.shape
    ##可以查看系数矩阵的内容
    print(x_test.toarray())
    st = x_test.toarray()
    
    fm = pylibfm.FM(num_factors=5, num_iter=500, verbose=True, task="classification",
                    initial_learning_rate=0.0001, learning_rate_schedule="optimal")
    
    fm.fit(x_train,y_train)
    

    ##模型预测
    y_score_test = fm.predict(x_test)
    y_pred = [1 if x >=0.5 else 0 for x in y_score_test ]
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
    ##计算fpr与tpr
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
   

    
    
    
    
    
    
    
    
