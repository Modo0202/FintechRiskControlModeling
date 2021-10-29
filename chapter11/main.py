# -*- coding: utf-8 -*-
"""
第11章：模型在线监控
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

def score_params_cal(base_point, odds, PDO):
    ##给定预期分数，与翻倍分数，确定参数A,B
    B = PDO/np.log(2)  
    A = base_point + B*np.log(odds)
    return A, B 
def myfunc(x):
    return str(x[0])+'_'+str(x[1])
##生成评分卡
def create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin):
    ##假设Odds在1:60时对应的参考分值为600分，分值调整刻度PDO为20，则计算得到分值转化的参数B = 28.85，A= 481.86。
    params_A,params_B = score_params_cal(base_point=600, odds=1/60, PDO=20)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
#        k='duration_BIN'
#        k = 'foreign_worker_BIN'
        if k !='intercept':
            df_temp =  pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin','woe_val']
            ##计算分值
            df_temp['score'] = round(-params_B*df_temp.woe_val*dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(df_temp['bin'],df_temp['score']))
            ##连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc,axis=1)
                df_1 = df_1[['total', 'var_name']]
                df_temp = pd.merge(df_temp , df_1,on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)
            ##离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                df_temp = pd.merge(df_temp , dict_disc_bin[k.split(sep='_BIN')[0]],on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)

    df_score['score_base'] =  base_points 
    return df_score,dict_bin_score,params_A,params_B,base_points
##计算样本分数
def cal_score(df_1,dict_bin_score,dict_cont_bin,dict_disc_bin,base_points):
    ##先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True,inplace = True)
    df_all_score = pd.DataFrame()
    ##连续变量
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([ df_all_score , varbin_meth.cont_var_bin_map(df_1[i], dict_cont_bin[i]).map(dict_bin_score[i]) ], axis = 1)
    ##离散变量
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([ df_all_score ,varbin_meth.disc_var_bin_map(df_1[i], dict_disc_bin[i]).map(dict_bin_score[i]) ], axis = 1)
    
    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points    
    df_all_score['score'] = df_all_score.apply(sum,axis=1)
    df_all_score['target'] = df_1.target
    return df_all_score

##计算整体PSI值
def cal_psi(df_raw, df_test,score_min,score_max,step):
    ##df_raw:pd.DataFrame训练集(线下数据)
    ##df_test:pd.DataFrame测试集(线上数据)
    score_bin = np.arange(score_min,score_max+step,step)
    total_raw = df_raw.shape[0]
    total_test = df_test.shape[0]
    psi = []
    total_all_raw = []
    total_all_test = []
    for i in range(len(score_bin)-1):
        total_1 = sum((df_raw.score >= score_bin[i]) & (df_raw.score < score_bin[i+1]))
        total_2 = sum((df_test.score >= score_bin[i]) & (df_test.score < score_bin[i+1]))
        if total_2==0:
            total_2 = 1
        if total_1==0:
            total_1 = 1
        psi.append( (total_1/total_raw - total_2/total_test )*(np.log((total_1/total_raw) / (total_2/total_test))) )
        total_all_raw.append(total_1)
        total_all_test.append(total_2)
    totle_psi = sum(psi)
    return totle_psi,total_all_raw,total_all_test
##计算单调性指标
def cal_kendall_tau(df_1,score_min,score_max,step,label='target'):
    score_bin = np.arange(score_min,score_max+step,step)
    bin_num = []
    for i in range(len(score_bin)-1):
        df_temp = df_1.loc[(df_1.score >= score_bin[i]) & (df_1.score < score_bin[i+1])]
        bin_num.append(df_temp[label].sum())
    concordant_pair = 0
    discordant_pair = 0
    for j in range(0, len(bin_num)-1):
        if bin_num[j] < bin_num[j+1]:
            discordant_pair += 1
        else:
            concordant_pair += 1
    ktau = (concordant_pair - discordant_pair) / (len(bin_num) * (len(bin_num) - 1) / 2)
    return ktau

if __name__ == '__main__':
    path = 'D:\\code\\chapter11'
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
        
   
    ########logistic模型
    ##参数优化
    lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    lr_gsearch.fit(x_train, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    
    ##最有参数训练模型
    LR_model_2 = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])
    
    LR_model_fit = LR_model_2.fit(x_train, y_train)
    
    ###保存模型的参数用于计算评分
    var_woe_name.append('intercept')
    weight_value = list(LR_model_fit.coef_.flatten())
    weight_value.extend(list(LR_model_fit.intercept_))
    dict_params = dict(zip(var_woe_name,weight_value))
     

    ##查看训练集、验证集与测试集
    y_score_train = LR_model_fit.predict_proba(x_train)[:, 1]
    y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
  
    ###看一下混沌矩阵
    y_pred = LR_model_fit.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 
    
    ####生成评分卡
    df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin)

    ##计算样本评分
    df_train_score = cal_score(data_train,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
    df_test_score = cal_score(data_test,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
    psi,total_raw,total_test = cal_psi(df_train_score, df_test_score,330,660,30)
    
    
    ##画图观察不同分数段内的样本分布
    val_1 = ['[330-360)','[360-390)','[390-420)','[420-450)','[450-480)','[480-510)','[510-540)',
    '[540-570)','[570-600]','[600-630]','[630-660]']
    plt.figure(figsize=(10,6))
    bar_width = 0.3
    fontsize_1 = 12
    plt.bar(np.arange(1,len(total_raw)+1),total_raw,bar_width,color='black', label='train_data')
    plt.bar(np.arange(1,len(total_raw)+1)+ bar_width,total_test,bar_width,color='grey', label='test_data')
    plt.xticks(np.arange(1,len(total_raw)+1),val_1, fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.xlabel('分值区间',fontsize=fontsize_1)
    plt.ylabel('频数',fontsize=fontsize_1)
    plt.legend(fontsize=fontsize_1)
    
    ktau_train = cal_kendall_tau(df_train_score,330,660,30,label='target')
    ktau_test = cal_kendall_tau(df_test_score,330,660,30,label='target')
    print('trainset ktau is {0},and testset  ktau is {1}'.format(ktau_train,
                 ktau_test)) 
        

