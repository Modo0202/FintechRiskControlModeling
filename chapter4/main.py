# -*- coding: utf-8 -*-
"""
第4章：数据清洗与预处理
"""
import os
import pandas as pd
import numpy as np
import time
import datetime
import missingno as msno

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

import warnings
warnings.filterwarnings("ignore") ##忽略警告

##数据读取
def data_read(data_path,file_name):

    df = pd.read_csv(os.path.join(data_path, file_name), delim_whitespace = True, header = None )
    ##变量重命名
    columns = ['status_account','duration','credit_history','purpose', 'amount',
               'svaing_account', 'present_emp', 'income_rate', 'personal_status',
               'other_debtors', 'residence_info', 'property', 'age',
               'inst_plans', 'housing', 'num_credits',
               'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    df.columns = columns
    ##将标签变量由状态1,2转为0,1;0表示好用户，1表示坏用户
    df.target = df.target - 1
    return df

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

def add_str(x):

    str_1 = ['%',' ','/t','$',';','@']
    str_2 = str_1[np.random.randint( 0,high = len(str_1)-1 )]
    return x+str_2

def add_time(num,style="%Y-%m-%d"):

    start_time = time.mktime((2010,1,1,0,0,0,0,0,0) )    
    stop_time = time.mktime((2015,1,1,0,0,0,0,0,0) )  
    re_time = []
    for i in range(num):
        rand_time = np.random.randint( start_time,stop_time)
        #将时间戳生成时间元组
        date_touple = time.localtime(rand_time)          
        re_time.append(time.strftime(style,date_touple))
    return re_time

def add_row(df_temp,num):
    
    index_1 = np.random.randint( low = 0,high = df_temp.shape[0]-1,size=num)
    return df_temp.loc[index_1]

if __name__ == '__main__':

    path = 'D:\\code\\chapter4'
    data_path = os.path.join(path ,'data')
    file_name = 'german.csv'
    ##读取数据
    df = data_read(data_path,file_name)
    ##区分离散变量与连续变量
    feature_names = list(df.columns)
    feature_names.remove('target')
    categorical_var,numerical_var = category_continue_separation(df,feature_names)
#    df.describe()
    ##########数据清洗################
    ##注入“脏数据”
    ##变量status_account随机加入特殊字符
    df.status_account = df.status_account.apply(add_str)
    ##添加两列时间格式的数据
    df['apply_time'] = add_time(df.shape[0],"%Y-%m-%d")
    df['job_time'] = add_time(df.shape[0],"%Y/%m/%d %H:%M:%S")
    ##添加行冗余数据
    df_temp = add_row(df,10)
    df = pd.concat([df,df_temp],axis=0,ignore_index=True)
    df.shape
    
    ###数据清洗
    ##默认值显示5列
    df.head()
    ##设置显示多列或全部全是
    pd.set_option('display.max_columns', 10)
    df.head()
    pd.set_option('display.max_columns', None)
    df.head()
    ##离散变量先看一下范围
    df.status_account.unique()
    
    ##特殊字符清洗
    df.status_account = df.status_account.apply(lambda x:x.replace(' ','').replace('%','').
                             replace('/t','').replace('$','').replace('@','').replace(';',''))
    df.status_account.unique()
    
    ##时间格式统一
    ##统一为'%Y-%m-%d格式
    df['job_time'] = df['job_time'].apply(lambda x:x.split(' ')[0].replace('/','-'))
    ##时间为字符串格式转为时间格式
    df['job_time'] = df['job_time'].apply(lambda x:datetime.datetime.strptime( x, '%Y-%m-%d'))
    df['apply_time'] = df['apply_time'].apply(lambda x:datetime.datetime.strptime( x, '%Y-%m-%d'))

    ##样本去冗余
    df.drop_duplicates(subset=None,keep='first',inplace=True)
    df.shape
    ##可以按照订单如冗余
    df['order_id'] = np.random.randint( low = 0,high = df.shape[0]-1,size=df.shape[0])
    df.drop_duplicates(subset=['order_id'],keep='first',inplace=True)
    df.shape
    ##如果有按列名去重复
#    df_1 = df.T
#    df_1 = df_1[~df_1.index.duplicated()]
#    df = df_1.T
    ###探索性分析
    df[numerical_var].describe()
    #添加缺失值
    df.reset_index(drop=True,inplace=True)
    var_name = categorical_var+numerical_var
    for i in var_name:
        num = np.random.randint( low = 0,high = df.shape[0]-1)
        index_1 = np.random.randint( low = 0,high = df.shape[0]-1,size=num)
        index_1 = np.unique(index_1)
        df[i].loc[index_1] = np.nan
        
    ##缺失值绘图
    msno.bar(df, labels=True,figsize=(10,6), fontsize=10)
    
    ##对于连续数据绘制箱线图，观察是否有异常值   
    plt.figure(figsize=(10,6))    #设置图形尺寸大小
    for j in range(1,len(numerical_var)+1):
        plt.subplot(2,4,j)
        df_temp = df[numerical_var[j-1]][~df[numerical_var[j-1]].isnull()]
        plt.boxplot( df_temp,
                    notch=False,  #中位线处不设置凹陷
                    widths=0.2,   #设置箱体宽度
                    medianprops={'color':'red'},  #中位线设置为红色
                    boxprops=dict(color="blue"),  #箱体边框设置为蓝色
                     labels=[numerical_var[j-1]],  #设置标签
                    whiskerprops = {'color': "black"}, #设置须的颜色，黑色
                    capprops = {'color': "green"},      #设置箱线图顶端和末端横线的属性，颜色为绿色
                    flierprops={'color':'purple','markeredgecolor':"purple"} #异常值属性，这里没有异常值，所以没表现出来
                   )
    plt.show()
    
    ####查看数据分布
    ##连续变量不同类别下的分布
    for i in numerical_var:
#        i = 'duration'
        ##取非缺失值的数据
        df_temp = df.loc[~df[i].isnull(),[i,'target']]
        df_good = df_temp[df_temp.target == 0]
        df_bad = df_temp[df_temp.target == 1]
        ##计算统计量
        valid = round(df_temp.shape[0]/df.shape[0]*100,2)
        Mean = round(df_temp[i].mean(),2)
        Std = round(df_temp[i].std(),2)
        Max = round(df_temp[i].max(),2)
        Min = round(df_temp[i].min(),2)
        ##绘图
        plt.figure(figsize=(10,6))
        fontsize_1 = 12
        plt.hist(df_good[i],  bins =20, alpha=0.5,label='好样本')
        plt.hist(df_bad[i],  bins =20, alpha=0.5,label='坏样本')
        plt.ylabel(i,fontsize=fontsize_1)
        plt.title( 'valid rate='+str(valid)+'%, Mean='+str(Mean) + ', Std='+str(Std)+', Max='+str(Max)+', Min='+str(Min))
        plt.legend()
        ##保存图片
        file = os.path.join(path,'plot_num', i+'.png')
        plt.savefig(file)
        plt.close(1)
        
    ##离散变量不同类别下的分布
    for i in categorical_var:
#        i = 'status_account'
        ##非缺失值数据
        df_temp = df.loc[~df[i].isnull(),[i,'target']]
        df_bad = df_temp[df_temp.target == 1]
        valid = round(df_temp.shape[0]/df.shape[0]*100,2)
        
        bad_rate = []
        bin_rate = []
        var_name = []
        for j in df[i].unique():
            
            if pd.isnull(j):
                df_1 = df[df[i].isnull()]
                bad_rate.append(sum(df_1.target)/df_1.shape[0])
                bin_rate.append(df_1.shape[0]/df.shape[0])
                var_name.append('NA')
            else:
                df_1 = df[df[i] == j]
                bad_rate.append(sum(df_1.target)/df_1.shape[0])
                bin_rate.append(df_1.shape[0]/df.shape[0])
                var_name.append(j)
        df_2 = pd.DataFrame({'var_name':var_name,'bin_rate':bin_rate,'bad_rate':bad_rate})
        ##绘图
        plt.figure(figsize=(10,6))
        fontsize_1 = 12
        plt.bar(np.arange(1,df_2.shape[0]+1),df_2.bin_rate,0.1,color='black',alpha=0.5, label='占比')
        plt.xticks(np.arange(1,df_2.shape[0]+1), df_2.var_name)
        plt.plot( np.arange(1,df_2.shape[0]+1),df_2.bad_rate,  color='green', alpha=0.5,label='坏样本比率')
        
        plt.ylabel(i,fontsize=fontsize_1)
        plt.title( 'valid rate='+str(valid)+'%')
        plt.legend()
        ##保存图片
        file = os.path.join(path,'plot_cat', i+'.png')
        plt.savefig(file)
        plt.close(1)
  















