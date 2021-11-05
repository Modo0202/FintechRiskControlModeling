# -*- coding: utf-8 -*-
"""
第6章：变量分箱方法
1:Chi-merge(卡方分箱), 2:IV(最优IV值分箱), 3:信息熵(基于树的分箱)
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

def cal_advantage(temp, piont, method,flag='sel'):
    """
    计算当前切分点下的指标值
    ##参数
    temp: 上一步的分箱结果，pandas dataframe
    piont: 切分点，以此来划分分箱
    method: 分箱方法选择，1:chi-merge , 2:IV值, 3:信息熵
    """
#    temp = binDS
    if flag == 'sel':
        ##用于最优切分点选择，这里只是二叉树，即二分
        bin_num = 2
        good_bad_matrix = np.empty((bin_num, 3))
        for ii in range(bin_num):
            if ii==0:
                df_temp_1 = temp[temp['bin_raw'] <= piont]
            else:
                df_temp_1 = temp[temp['bin_raw'] > piont]
            ##计算每个箱内的好坏样本书
            good_bad_matrix[ii][0] = df_temp_1['good'].sum()
            good_bad_matrix[ii][1] = df_temp_1['bad'].sum()
            good_bad_matrix[ii][2] = df_temp_1['total'].sum()
                    
    
    elif flag == 'gain':
       ##用于计算本次分箱后的指标结果，即分箱数，每增加一个，就要算一下当前分箱下的指标结果
       bin_num = temp['bin'].max()
       good_bad_matrix = np.empty((bin_num, 3))
       for ii in range(bin_num):
           df_temp_1 = temp[temp['bin'] == (ii + 1)]
           good_bad_matrix[ii][0] = df_temp_1['good'].sum()
           good_bad_matrix[ii][1] = df_temp_1['bad'].sum()
           good_bad_matrix[ii][2] = df_temp_1['total'].sum()
       
    ##计算总样本中的好坏样本
    total_matrix = np.empty(3)
    total_matrix[0] = temp.good.sum()
    total_matrix[1] = temp.bad.sum()
    total_matrix[2] = temp.total.sum()
    
    # Chi-merger分箱
    if method == 1:
        X2 = 0
        for i in range(bin_num):
            for j in range(2):
                expect = (total_matrix[j] / total_matrix[2])*good_bad_matrix[i][2]
                X2 = X2 + (good_bad_matrix[i][j] - expect )**2/expect
        M_value = X2
    # IV分箱
    elif method == 2:
        if pd.isnull(total_matrix[0]) or  pd.isnull(total_matrix[1]) or total_matrix[0] == 0 or total_matrix[1] == 0:
            M_value = np.NaN
        else:
            IV = 0
            for i in range(bin_num):
                ##坏好比
                weight = good_bad_matrix[i][1] / total_matrix[1] - good_bad_matrix[i][0] / total_matrix[0]
                IV = IV + weight * np.log( (good_bad_matrix[i][1] * total_matrix[0]) / (good_bad_matrix[i][0] * total_matrix[1]))
            M_value = IV
    # 信息熵分箱
    elif method == 3:
        ##总的信息熵    
        entropy_total = 0
        for j in range(2):
            weight = (total_matrix[j]/ total_matrix[2])
            entropy_total = entropy_total - weight * (np.log(weight))
                    
        ##计算条件熵
        entropy_cond = 0
        for i in range(bin_num):
            entropy_temp = 0
            for j in range(2):
                entropy_temp = entropy_temp - ((good_bad_matrix[i][j] / good_bad_matrix[i][2]) \
                                         * np.log(good_bad_matrix[i][j] / good_bad_matrix[i][2]) )
            entropy_cond = entropy_cond + good_bad_matrix[i][2]/total_matrix[2] * entropy_temp 
        
        ##计算归一化信息增益   
        M_value = 1 - (entropy_cond / entropy_total)  
    # Best-Ks分箱
    else:
        pass
    return M_value

def best_split(df_temp0, method, bin_num):
    """
    在每个候选集中寻找切分点，完成一次分裂。
    select_split_point函数的中间过程函数
    ##参数
    df_temp0: 上一次分箱后的结果，pandas dataframe
    method: 分箱方法选择，1:chi-merge , 2:IV值, 3:信息熵
    bin_num: 分箱编号，在不同编号的分箱结果中继续二分
    ##返回值
    返回在本次分箱标号内的最有切分结果， pandas dataframe
    """
#    df_temp0 = df_temp
#    bin_num = 1
    df_temp0 = df_temp0.sort_values(by=['bin', 'bad_rate'])
    piont_len = len(df_temp0[df_temp0['bin'] == bin_num])  ##候选集的长度
    bestValue = 0
    bestI = 1
    ##以候选集的每个切分点做分隔，计算指标值
    for i in range(1, piont_len):
        #计算指标值 
        value = cal_advantage(df_temp0,i,method,flag='sel')
        if bestValue < value:
            bestValue = value
            bestI = i
    # create new var split
    df_temp0['split'] = np.where(df_temp0['bin_raw'] <= bestI, 1, 0)
    df_temp0 = df_temp0.drop('bin_raw', axis=1)
    newbinDS = df_temp0.sort_values(by=['split', 'bad_rate'])
    # rebuild var i
    newbinDS_0 = newbinDS[newbinDS['split'] == 0]
    newbinDS_1 = newbinDS[newbinDS['split'] == 1]
    newbinDS_0 = newbinDS_0.copy()
    newbinDS_1 = newbinDS_1.copy()
    newbinDS_0['bin_raw'] = range(1, len(newbinDS_0) + 1)
    newbinDS_1['bin_raw'] = range(1, len(newbinDS_1) + 1)
    newbinDS = pd.concat([newbinDS_0, newbinDS_1], axis=0)
    return newbinDS  


def select_split_point(temp_bin, method):
    """
    二叉树分割方式，从候选者中挑选每次的最优切分点，与切分后的指标计算
    cont_var_bin函数的中间过程函数，
    ##参数
    temp_bin: 分箱后的结果 pandas dataframe
    method:分箱方法选择，1:chi-merge , 2:IV值, 3:信息熵
    ##返回值
    新的分箱结果  pandas dataframe
    """
#    temp_bin = df_temp_all
    temp_bin = temp_bin.sort_values(by=['bin', 'bad_rate'])
    ##得到最大的分箱值
    max_num = max(temp_bin['bin'])
#    temp_binC = dict()
#    m = dict()
#    ##不同箱内的数据取出来
#    for i in range(1, max_num + 1):
#        temp_binC[i] = temp_bin[temp_bin['bin'] == i]
#        m[i] = len(temp_binC[i])
    temp_main = dict()
    bin_i_value = []
    for i in range(1, max_num + 1):
        df_temp = temp_bin[temp_bin['bin'] == i]
        if df_temp.shape[0]>1 : 
            ##bin=i的做分裂
            temp_split= best_split(df_temp, method, i)
            ##完成一次分箱，更新bin的之
            temp_split['bin'] = np.where(temp_split['split'] == 1,
                                               max_num + 1,
                                               temp_split['bin'])
            ##取出bin!=i合并为新租
            temp_main[i] = temp_bin[temp_bin['bin'] != i]
            temp_main[i] = pd.concat([temp_main[i], temp_split ], axis=0, sort=False)
            ##计算新分组的指标值
            value = cal_advantage(temp_main[i],0, method,flag='gain')
            newdata = [i, value]
            bin_i_value.append(newdata)
    # find maxinum of value bintoSplit
    bin_i_value.sort(key=lambda x: x[1], reverse=True)
    # binNum = temp_all_Vals['BinToSplit']
    binNum = bin_i_value[0][0]
    newBins = temp_main[binNum].drop('split', axis=1)
    return newBins.sort_values(by=['bin', 'bad_rate']), round( bin_i_value[0][1] ,4)


def init_equal_bin(x,bin_rate):
    """
    初始化等距分组，cont_var_bin函数的中间过程函数
    ##参数
    x:要分组的变量值，pandas series
    bin_rate：比例值1/bin_rate
    ##返回值
    返回初始化分箱结果，pandas dataframe
    """
    ##异常值剔除，只考虑90%没的最大值与最小值，边界与-inf或inf分为一组
    if len(x[x > np.percentile(x, 95)]) > 0 and len(np.unique(x)) >=30:
        var_up= min( x[x > np.percentile(x, 95)] )
    else:
        var_up = max(x)
    if len(x[x < np.percentile(x, 5)]) > 0:
        var_low= max( x[x < np.percentile(x, 5)] )
    else:
        var_low = min(x)
    ##初始化分组
    bin_num = int(1/ bin_rate)
    dist_bin = (var_up - var_low) / bin_num  ##分箱间隔
    bin_up = []
    bin_low = []
    for i in range(1, bin_num + 1):
        if i == 1:
            bin_up.append( var_low + i * dist_bin)
            bin_low.append(-np.inf)
        elif i == bin_num:
            bin_up.append( np.inf)
            bin_low.append( var_low + (i - 1) * dist_bin )
        else:
            bin_up.append( var_low + i * dist_bin )
            bin_low.append( var_low + (i - 1) * dist_bin )
    result = pd.DataFrame({'bin_up':bin_up,'bin_low':bin_low})
    result.index.name = 'bin_num'
    return result
def limit_min_sample(temp_cont,  bin_min_num_0):
    """
    分箱约束条件：每个箱内的样本数不能小于bin_min_num_0，cont_var_bin函数的中间过程函数
    ##参数
    temp_cont: 初始化分箱后的结果 pandas dataframe
    bin_min_num_0:每组内的最小样本限制
    ##返回值
    合并后的分箱结果，pandas dataframe
    """
    for i in temp_cont.index:
        rowdata = temp_cont.loc[i, :]
        if i == temp_cont.index.max():
            ##如果是最后一个箱就，取倒数第二个值
            ix = temp_cont[temp_cont.index < i].index.max()
        else:
            ##否则就取大于i的最小的分箱值
            ix = temp_cont[temp_cont.index > i].index.min()
        ##如果0, 1, total项中样本的数量小于20则进行合并
        if rowdata['total'] <= bin_min_num_0:
            # 与相邻的bin合并
            temp_cont.loc[ix, 'bad'] = temp_cont.loc[ix, 'bad'] + rowdata['bad']
            temp_cont.loc[ix, 'good'] = temp_cont.loc[ix, 'good'] + rowdata['good']
            temp_cont.loc[ix, 'total'] = temp_cont.loc[ix, 'total'] + rowdata['total']
            if i < temp_cont.index.max():
                temp_cont.loc[ix, 'bin_low'] = rowdata['bin_low']
            else:
                temp_cont.loc[ix, 'bin_up'] = rowdata['bin_up']
            temp_cont = temp_cont.drop(i, axis=0)  
    return temp_cont.sort_values(by='bad_rate')
def cont_var_bin_map(x, bin_init):
    """
    按照初始化分箱结果，对原始值进行分箱映射
    用于训练集与测试集的分箱映射
    """
    temp = x.copy()
    for i in bin_init.index:
        bin_up = bin_init['bin_up'][i]
        bin_low = bin_init['bin_low'][i]
        # 寻找出 >lower and <= upper的位置
        if pd.isnull(bin_up) or pd.isnull(bin_up):
            temp[pd.isnull(temp)] = i
        else:
            index = (x > bin_low) & (x <= bin_up)
            temp[index] = i
    temp.name = temp.name + "_BIN"
    return temp
def merge_bin(sub, i):
    """
    将相同箱内的样本书合并，区间合并
    ##参数
    sub:分箱结果子集，pandas dataframe ，如bin=1的结果
    i: 分箱标号
    ##返回值
    返回合并结果
    """
    l = len(sub)
    total = sub['total'].sum()
    first = sub.iloc[0, :]
    last = sub.iloc[l - 1, :]

    lower = first['bin_low']
    upper = last['bin_up']
    df = pd.DataFrame()
    df = df.append([i, lower, upper, total], ignore_index=True).T
    df.columns = ['bin', 'bin_low', 'bin_up', 'total']
    return df


def cont_var_bin(x, y, method, mmin=5, mmax=10, bin_rate=0.01, stop_limit=0.1, bin_min_num=20):
    """
    ##连续变量分箱函数
    ##参数
    x:输入分箱数据，pandas series
    y:标签变量
    method:分箱方法选择，1:chi-merge , 2:IV值, 3:基尼系数分箱
    mmin:最小分箱数，当分箱初始化后如果初始化箱数小于等mmin，则mmin=2，即最少分2箱，
         如果分两箱也无法满足箱内最小样本数限制而分1箱，则变量删除
    mmax:最大分箱数，当分箱初始化后如果初始化箱数小于等于mmax，则mmax等于初始化箱数-1
    bin_rate：等距初始化分箱参数，分箱数为1/bin_rate,分箱间隔在数据中的最小值与最大值将等间隔取值
    stop_limit:分箱earlystopping机制，如果已经没有明显增益即停止分箱
    bin_min_num:每组最小样本数
    ##返回值
    分箱结果：pandas dataframe
    """
#    x= data_train.amount
#    y = data_train.target
#    method=2
#    mmin=4
#    mmax=10
#    bin_rate=0.01
#    stop_limit=0.05
#    bin_min_num=5

    ##缺失值单独取出来
    df_na = pd.DataFrame({'x': x[pd.isnull(x)], 'y': y[pd.isnull(x)]})
    y = y[~pd.isnull(x)]
    x = x[~pd.isnull(x)]
    ##初始化分箱，等距的方式，后面加上约束条件,没有箱内样本数没有限制
    bin_init = init_equal_bin(x, bin_rate)
    ##分箱映射
    bin_map = cont_var_bin_map(x, bin_init)
    
    df_temp = pd.concat([x, y, bin_map], axis=1)
    ##计算每个bin中好坏样本的频数
    df_temp_1 = pd.crosstab(index=df_temp[bin_map.name], columns=y)
    df_temp_1.rename(columns= dict(zip([0,1], ['good', 'bad'])) , inplace=True)
    ##计算每个bin中一共有多少样本
    df_temp_2 = pd.DataFrame(df_temp.groupby(bin_map.name).count().iloc[:, 0])
    df_temp_2.columns = ['total']
    df_temp_all= pd.merge(pd.concat([df_temp_1, df_temp_2], axis=1), bin_init,
                         left_index=True, right_index=True,
                         how='left')
    
    ####做分箱上下限的整理，让候选点连续
    for j in range(df_temp_all.shape[0]-1):
        if df_temp_all.bin_low.loc[df_temp_all.index[j+1]] !=  df_temp_all.bin_up.loc[df_temp_all.index[j]]:
            df_temp_all.bin_low.loc[df_temp_all.index[j+1]] = df_temp_all.bin_up.loc[df_temp_all.index[j]]
        
    ##离散变量中这个值为badrate,连续变量时为索引，索引值是分箱初始化时，箱内有变量的箱的索引
    df_temp_all['bad_rate'] = df_temp_all.index
    ##最小样本数限制，进行分箱合并
    df_temp_all = limit_min_sample(df_temp_all, bin_min_num)
    ##将合并后的最大箱数与设定的箱数进行比较，这个应该是分箱数的最大值
    if mmax >= df_temp_all.shape[0]:
        mmax = df_temp_all.shape[0]-1
    if mmin >= df_temp_all.shape[0]:
        gain_value_save0=0
        gain_rate_save0=0
        df_temp_all['bin'] = np.linspace(1,df_temp_all.shape[0],df_temp_all.shape[0],dtype=int)
        data = df_temp_all[['bin_low','bin_up','total','bin']]
        data.index = data['bin']
    else:
        df_temp_all['bin'] = 1
        df_temp_all['bin_raw'] = range(1, len(df_temp_all) + 1)
        df_temp_all['var'] = df_temp_all.index  ##初始化箱的编号
        gain_1 = 1e-10
        gain_rate_save0 = []
        gain_value_save0 = []
        ##分箱约束：最大分箱数限制
        for i in range(1,mmax):
    #        i = 1
            df_temp_all, gain_2 = select_split_point(df_temp_all, method=method)
            gain_rate = gain_2 / gain_1 - 1  ## ratio gain
            gain_value_save0.append(np.round(gain_2,4))
            if i == 1:
                gain_rate_save0.append(0.5)
            else:
                gain_rate_save0.append(np.round(gain_rate,4))
            gain_1 = gain_2
            if df_temp_all.bin.max() >= mmin and df_temp_all.bin.max() <= mmax:
                if gain_rate <= stop_limit or pd.isnull(gain_rate):
                    break
                
    
        df_temp_all = df_temp_all.rename(columns={'var': 'oldbin'})
        temp_Map1 = df_temp_all.drop(['good', 'bad', 'bad_rate', 'bin_raw'], axis=1)
        temp_Map1 = temp_Map1.sort_values(by=['bin', 'oldbin'])
        # get new lower, upper, bin, total for sub
        data = pd.DataFrame()
        for i in temp_Map1['bin'].unique():
            ##得到这个箱内的上下界
            sub_Map = temp_Map1[temp_Map1['bin'] == i]
            rowdata = merge_bin(sub_Map, i)
            data = data.append(rowdata, ignore_index=True)
    
        # resort data
        data = data.sort_values(by='bin_low')
        data = data.drop('bin', axis=1)
        mmax = df_temp_all.bin.max()
        data['bin'] = range(1, mmax + 1)
        data.index = data['bin']
    ##将缺失值的箱加过来
    if len(df_na) > 0:
        row_num = data.shape[0] + 1
        data.loc[row_num, 'bin_low'] = np.nan
        data.loc[row_num, 'bin_up'] = np.nan
        data.loc[row_num, 'total'] = df_na.shape[0]
        data.loc[row_num, 'bin'] = data.bin.max() + 1
    return data , gain_value_save0 ,gain_rate_save0


def cal_bin_value(x, y, bin_min_num_0=10):
    """
    按变量类别进行分箱初始化，不满足最小样本数的箱进行合并
    ##参数
    x: 待分箱的离散变量 pandas Series
    y: 标签变量
    target: 正样本标识
    bin_min_num_0：箱内的最小样本数限制
    ##返回值
    计算结果
    """
    ##按类别x计算yz中0,1两种状态的样本数
    df_temp = pd.crosstab(index=x, columns=y, margins=False)
    df_temp.rename(columns= dict(zip([0,1], ['good', 'bad'])) , inplace=True)
    df_temp = df_temp.assign(total=lambda x:x['good']+ x['bad'],bin=1,var_name=df_temp.index).assign(bad_rate=lambda x:x['bad']/ x['total'])

    ##按照baterate排序
    df_temp = df_temp.sort_values(by='bad_rate')
    df_temp = df_temp.reset_index(drop=True)
    ##样本数不满足最小值进行合并
    for i in df_temp.index:
        rowdata = df_temp.loc[i, :]
        if i == df_temp.index.max():
            ##如果是最后一个箱就，取倒数第二个值
            ix = df_temp[df_temp.index < i].index.max()
        else:
            ##否则就取大于i的最小的分箱值
            ix = df_temp[df_temp.index > i].index.min()
        ##如果0, 1, total项中样本的数量小于20则进行合并
        if any(rowdata[:3] <= bin_min_num_0):
            # 与相邻的bin合并
            df_temp.loc[ix, 'bad'] = df_temp.loc[ix, 'bad'] + rowdata['bad']
            df_temp.loc[ix, 'good'] = df_temp.loc[ix, 'good'] + rowdata['good']
            df_temp.loc[ix, 'total'] = df_temp.loc[ix, 'total'] + rowdata['total']
            df_temp.loc[ix, 'bad_rate'] = df_temp.loc[ix,'bad'] / df_temp.loc[ix, 'total']
            # 将区间也进行合并
            df_temp.loc[ix, 'var_name'] = str(rowdata['var_name']) +'%'+ str(df_temp.loc[ix, 'var_name'])
         
            df_temp = df_temp.drop(i, axis=0)  ##删除原来的bin
    ##如果离散变量小于等于5，每个变量为一个箱
    df_temp['bin_raw'] = range(1, df_temp.shape[0] + 1)
    df_temp = df_temp.reset_index(drop=True)
    return df_temp


def disc_var_bin(x, y, method=1, mmin=3, mmax=8, stop_limit=0.1, bin_min_num = 20  ):
    """
    离散变量分箱方法，如果变量过于稀疏最好先编码在按连续变量分箱
    ##参数：
    x:输入分箱数据，pandas series
    y:标签变量
    method:分箱方法选择，1:chi-merge , 2:IV值, 3:信息熵
    mmin:最小分箱数，当分箱初始化后如果初始化箱数小于等mmin，则mmin=2，即最少分2箱，
         如果分两厢也无法满足箱内最小样本数限制而分1箱，则变量删除
    mmax:最大分箱数，当分箱初始化后如果初始化箱数小于等于mmax，则mmax等于初始化箱数-1
    stop_limit:分箱earlystopping机制，如果已经没有明显增益即停止分箱
    bin_min_num:每组最小样本数
    ##返回值
    分箱结果：pandas dataframe
    """
#    x = data_train.purpose
#    y = data_train.target
    del_key = []    
    ##缺失值单独取出来
    df_na = pd.DataFrame({'x': x[pd.isnull(x)], 'y': y[pd.isnull(x)]})
    y = y[~pd.isnull(x)]
    x = x[~pd.isnull(x)]
    ##数据类型转化
    if np.issubdtype(x.dtype, np.int_):
        x = x.astype('float').astype('str')
    if np.issubdtype(x.dtype, np.float_):
        x = x.astype('str')
  
    ##按照类别分箱，得到每个箱下的统计值
    temp_cont = cal_bin_value(x, y,bin_min_num)
    
    ##如果去掉缺失值后离散变量的可能取值小于等于5不分箱
    if len(x.unique()) > 5:
        ##将合并后的最大箱数与设定的箱数进行比较，这个应该是分箱数的最大值
        if mmax >= temp_cont.shape[0]:
            mmax = temp_cont.shape[0]-1
        if mmin >= temp_cont.shape[0]:
            mmin = 2
            mmax = temp_cont.shape[0]-1
        if mmax ==1:
            print('变量 {0}合并后分箱数为1，该变量删除'.format(x.name))
            del_key.append(x.name)
        
        gain_1 = 1e-10
        gain_value_save0 = []
        gain_rate_save0 = []
        for i in range(1,mmax):
            temp_cont, gain_2 = select_split_point(temp_cont, method=method)
            gain_rate = gain_2 / gain_1 - 1  ## ratio gain
            gain_value_save0.append(np.round(gain_2,4))
            if i == 1:
                gain_rate_save0.append(0.5)
            else:
                gain_rate_save0.append(np.round(gain_rate,4))
            gain_1 = gain_2
            if temp_cont.bin.max() >= mmin and temp_cont.bin.max() <= mmax:
                if gain_rate <= stop_limit:
                    break
    
        temp_cont = temp_cont.rename(columns={'var': x.name})
        temp_cont = temp_cont.drop(['good', 'bad', 'bin_raw', 'bad_rate'], axis=1)
    else:
        temp_cont.bin = temp_cont.bin_raw
        temp_cont = temp_cont[['total', 'bin', 'var_name']]
        gain_value_save0=[]
        gain_rate_save0=[]
        del_key=[]
    ##将缺失值的箱加过来
    if len(df_na) > 0:
        index_1 = temp_cont.shape[0] + 1
        temp_cont.loc[index_1, 'total'] = df_na.shape[0]
        temp_cont.loc[index_1, 'bin'] = temp_cont.bin.max() + 1
        temp_cont.loc[index_1, 'var_name'] = 'NA'
    temp_cont = temp_cont.reset_index(drop=True)  
    if temp_cont.shape[0]==1:
        del_key.append(x.name)
    return temp_cont.sort_values(by='bin') , gain_value_save0 , gain_rate_save0,del_key


def disc_var_bin_map(x, bin_map):
    """
    用离散变量分箱后的结果，对原始值进行分箱映射
    ##参数
    x: 待分箱映射的离散变量，pandas Series
    bin_map:分箱映射字典， pandas dataframe
    ##返回值
    返回映射结果
    """
    ##数据类型转化
    xx = x[~pd.isnull(x)]
    if np.issubdtype(xx.dtype, np.int_):
        x[~pd.isnull(x)] = xx.astype('float').astype('str')
    if np.issubdtype(xx.dtype, np.float_):
        x[~pd.isnull(x)] = xx.astype('str') 
    d = dict()
    for i in bin_map.index:
        for j in  bin_map.loc[i,'var_name'].split('%'):
            if j != 'NA':
                d[j] = bin_map.loc[i,'bin']

    new_x = x.map(d)
    ##有缺失值要做映射
    if sum(pd.isnull(new_x)) > 0:
        index_1 = bin_map.index[bin_map.var_name == 'NA']
        if len(index_1) > 0:
            new_x[pd.isnull(new_x)] = bin_map.loc[index_1,'bin'].tolist()
    new_x.name = x.name + '_BIN'

    return new_x

if __name__ == '__main__':
    
    path = 'D:/code_1/chapter6/'
    data_path = os.path.join(path,'data')
    file_name = 'german.csv'
    ##读取数据
    data_train, data_test = data_read(data_path,file_name)
    ##连续变量分箱
    data_train.amount[1:30] = np.nan
    data_test1,gain_value_save1 ,gain_rate_save1  = cont_var_bin(data_train.amount, data_train.target, 
                             method=1, mmin=4 ,mmax=10,bin_rate=0.01,stop_limit=0.1 ,bin_min_num=20 )
    
    data_test2,gain_value_save2 ,gain_rate_save2  = cont_var_bin(data_train.amount, data_train.target,
                             method=2, mmin=4 ,mmax=10,bin_rate=0.01,stop_limit=0.1 ,bin_min_num=20 )

    data_test3,gain_value_save3 ,gain_rate_save3 = cont_var_bin(data_train.amount, data_train.target, 
                             method=3, mmin=4 ,mmax=10,bin_rate=0.01,stop_limit=0.1 ,bin_min_num=20 )
    
   
    ###区分离散变量和连续变量批量进行分箱，把每个变量分箱的结果保存在字典中
    dict_cont_bin = {}
    cont_name = ['duration', 'amount', 'income_rate',  'residence_info',  
               'age',  'num_credits','dependents']
    for i in cont_name:
        dict_cont_bin[i],gain_value_save , gain_rate_save = cont_var_bin(data_train[i], data_train.target, method=1, mmin=4, mmax=10,
                                     bin_rate=0.01, stop_limit=0.1, bin_min_num=20)

    ##离散变量分箱
    ##离散变量分箱
    data_train.purpose[1:30] = np.nan
    data_disc_test1,gain_value_save1 ,gain_rate_save1,del_key  = disc_var_bin(data_train.purpose, data_train.target, 
                             method=1, mmin=4 ,mmax=10,stop_limit=0.1 ,bin_min_num=10 )
    
    data_disc_test2,gain_value_save2 ,gain_rate_save2 ,del_key = disc_var_bin(data_train.purpose, data_train.target,
                             method=2, mmin=4 ,mmax=10,stop_limit=0.1 ,bin_min_num=10 )

    data_disc_test3,gain_value_save3 ,gain_rate_save3,del_key = disc_var_bin(data_train.purpose, data_train.target, 
                             method=3, mmin=4 ,mmax=10,stop_limit=0.1 ,bin_min_num=10 )
    
    dict_disc_bin = {}
    del_key = []
    disc_name = [x for x in data_train.columns if x not in cont_name]
    disc_name.remove('target')
    for i in disc_name:
        dict_disc_bin[i],gain_value_save , gain_rate_save,del_key_1  = disc_var_bin(data_train[i], data_train.target, method=1, mmin=3,
                                     mmax=8, stop_limit=0.1, bin_min_num=5)
        if len(del_key_1)>0 :
            del_key.extend(del_key_1)
    ###删除分箱数只有1个的变量
    if len(del_key) > 0:
        for j in del_key:
            del dict_disc_bin[j]
    ##训练数据分箱
    ##连续变量分箱映射
#    ss = data_train[list( dict_cont_bin.keys())]
    df_cont_bin_train = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_train = pd.concat([ df_cont_bin_train , cont_var_bin_map(data_train[i], dict_cont_bin[i]) ], axis = 1)
    ##离散变量分箱映射
#    ss = data_train[list( dict_disc_bin.keys())]
    df_disc_bin_train = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_train = pd.concat([ df_disc_bin_train , disc_var_bin_map(data_train[i], dict_disc_bin[i]) ], axis = 1)

    ##测试数据分箱
    ##连续变量分箱映射
    ss = data_test[list( dict_cont_bin.keys())]
    df_cont_bin_test = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_test = pd.concat([ df_cont_bin_test , cont_var_bin_map(data_test[i], dict_cont_bin[i]) ], axis = 1)
    ##离散变量分箱映射
#    ss = data_test[list( dict_disc_bin.keys())]
    df_disc_bin_test = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_test = pd.concat([ df_disc_bin_test , disc_var_bin_map(data_test[i], dict_disc_bin[i]) ], axis = 1)














