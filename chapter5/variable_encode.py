# -*- coding: utf-8 -*-
"""
第5章 变量编码
变量编码:one-hot编码、标签编码、自定义字典映射、woe编码
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore") ##忽略警告


def data_read(data_path,file_name):
    df = pd.read_csv(os.path.join(data_path, file_name), delim_whitespace=True, header=None)
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

##one—hot编码
def onehot_encode(df,data_path_1,flag='train'):
    df = df.reset_index(drop=True)
    ##判断数据集是否存在缺失值
    if sum(df.isnull().any()) > 0 :
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        var_numerics = df.select_dtypes(include=numerics).columns
        var_str = [ i for i in df.columns if i not in  var_numerics ]
        ##数据类型的缺失值用-77777填补
        if len(var_numerics) > 0:
            df.loc[:,var_numerics] = df[var_numerics].fillna(-7777)
        ##字符串类型的缺失值用NA填补
        if len(var_str) > 0:
            df.loc[:,var_str] = df[var_str].fillna('NA')
            
    if flag == 'train':
        enc = OneHotEncoder(dtype='int').fit(df)
        ##保存编码模型
        save_model = open(os.path.join(data_path_1, 'onehot.pkl'), 'wb')
        pickle.dump(enc, save_model, 0)
        save_model.close()
        df_return = pd.DataFrame( enc.transform(df).toarray())
        df_return.columns = enc.get_feature_names(df.columns)
        
    elif flag =='test':
        ##测试数据编码
        read_model = open(os.path.join(data_path_1, 'onehot.pkl'), 'rb')
        onehot_model = pickle.load(read_model)
        read_model.close()
        ##如果训练集无缺失值，测试集有缺失值则将该样本删除
        var_range = onehot_model.categories_
        var_name = df.columns
        del_index = []
        for i in range(len(var_range)):
            if 'NA' not in var_range[i]and 'NA' in df[var_name[i]].unique():
                index = np.where( df[var_name[i]] == 'NA')
                del_index.append(index)
            elif -7777 not in var_range[i] and -7777 in df[var_name[i]].unique():
                index = np.where( df[var_name[i]] == -7777)
                del_index.append(index)
        ##删除样本
        if len(del_index) > 0:
            del_index = np.unique(del_index)
            df = df.drop(del_index)
            print('训练集无缺失值，但测试集有缺失值，第{0}条样本被删除'.format(del_index))
        df_return = pd.DataFrame(onehot_model.transform(df).toarray())
        df_return.columns = onehot_model.get_feature_names(df.columns)
        
    elif flag == 'transform':
        ##编码数据值转化为原始变量
        read_model = open(os.path.join(data_path_1,'onehot.pkl'),'rb')
        onehot_model = pickle.load(read_model)
        read_model.close()
        ##逆变换
        df_return = pd.DataFrame(onehot_model.inverse_transform(df))
        df_return.columns = np.unique(['_'.join(i.rsplit('_')[:-1]) for i in df.columns])
    return df_return

##标签编码
def label_encode(df,data_path_1,flag='train'):
    if flag == 'train':
        enc = LabelEncoder().fit(df)
        ##保存编码模型
        save_model = open(os.path.join(data_path_1, 'labelcode.pkl'), 'wb')
        pickle.dump(enc, save_model, 0)
        save_model.close()
        df_return = pd.DataFrame( enc.transform(df))
        df_return.name = df.name
        
    elif flag =='test':
        ##测试数据编码
        read_model = open(os.path.join(data_path_1, 'labelcode.pkl'), 'rb')
        label_model = pickle.load(read_model)
        read_model.close()
        df_return = pd.DataFrame(label_model.transform(df))
        df_return.name = df.name

    elif flag == 'transform':
        ##编码数据值转化为原始变量
        read_model = open(os.path.join(data_path_1, 'labelcode.pkl'), 'rb')
        label_model = pickle.load(read_model)
        read_model.close()
        ##逆变换
        df_return = pd.DataFrame(label_model.inverse_transform(df))
    return df_return

def dict_encode(df, data_path_1):
    ##自定义映射
    embarked_mapping = {}
    embarked_mapping['status_account'] = {'NA': 1, 'A14': 2, 'A11':3,'A12': 4,'A13':5}  
    embarked_mapping['svaing_account'] = {'NA': 1, 'A65': 1, 'A61':3,'A62': 5,'A63':6,'A64':8}  
    embarked_mapping['present_emp'] = {'NA': 1, 'A71': 2, 'A72':5,'A73': 6,'A74':8,'A75':10}  
    embarked_mapping['property'] = {'NA': 1, 'A124': 1, 'A123':4,'A122': 6, 'A121':9 } 

    df = df.reset_index(drop=True)
    ##判断数据集是否存在缺失值
    if sum(df.isnull().any()) > 0 :
        df = df.fillna('NA')
    ##字典映射
    var_dictEncode = []        
    for i in df.columns:
        col = i + '_dictEncode'
        df[col] = df[i].map(embarked_mapping[i])
        var_dictEncode.append(col)
    return df[var_dictEncode]
    
##WOE编码
def woe_cal_trans(x, y, target=1):
    ##计算总体的正负样本数
    p_total = sum(y == target)
    n_total = len(x)-p_total
    value_num = list(x.unique())
    woe_map = {}
    iv_value = 0
    for i in value_num:
        ##计算该变量取值箱内的正负样本总数    
        y1 = y[np.where(x == i)[0]]
        p_num_1 = sum(y1 == target)
        n_num_1 = len(y1) - p_num_1
        ##计算占比
        bad_1 = p_num_1 / p_total
        good_1 =  n_num_1 / n_total
        if bad_1 == 0:
            bad_1 = 1e-5
        elif good_1 == 0:
            good_1 = 1e-5
        woe_map[i] = np.log(bad_1 / good_1)
        iv_value += (bad_1 - good_1) * woe_map[i]
    x_woe_trans = x.map(woe_map)
    x_woe_trans.name = x.name + "_woe"
    return x_woe_trans, woe_map, iv_value

def woe_encode(df, data_path_1, varnames, y, filename, flag='train'):
    """
    WOE编码映射
    ---------------------------------------
    Param
    df: pandas dataframe,待编码数据
    data_path_1 :存取文件路径
    varnames: 变量列表
    y:  目标变量
    filename:编码存取的文件名
    flag: 选择训练还是测试
    ---------------------------------------
    Return
    df: pandas dataframe, 编码后的数据，包含了原始数据
    woe_maps: dict,woe编码字典
    iv_values: dict, 每个变量的IV值
    """  
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    ##判断数据集是否存在缺失值
    if sum(df.isnull().any()) > 0 :
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        var_numerics = df.select_dtypes(include=numerics).columns
        var_str = [ i for i in df.columns if i not in  var_numerics ]
        ##数据类型的缺失值用-77777填补
        if len(var_numerics) > 0:
            df.loc[:,var_numerics] = df[var_numerics].fillna(-7777)
        ##字符串类型的缺失值用NA填补
        if len(var_str) > 0:
            df.loc[:,var_str] = df[var_str].fillna('NA')
            
    if flag == 'train':
        iv_values = {}
        woe_maps = {}
        var_woe_name = []
        for var in varnames:
            x = df[var]
            ##变量映射
            x_woe_trans, woe_map, info_value = woe_cal_trans(x, y)
            var_woe_name.append(x_woe_trans.name)
            df = pd.concat([df, x_woe_trans], axis=1)
            woe_maps[var] = woe_map
            iv_values[var] = info_value
        ##保存woe映射字典
        save_woe_dict = open(os.path.join(data_path_1, filename + '.pkl'), 'wb')
        pickle.dump(woe_maps, save_woe_dict, 0)
        save_woe_dict.close()
        return df, woe_maps, iv_values, var_woe_name
        
    elif flag == 'test':
         ##测试数据编码
         read_woe_dict = open(os.path.join(data_path_1, filename + '.pkl'), 'rb')
         woe_dict = pickle.load(read_woe_dict)
         read_woe_dict.close()
         ##如果训练集无缺失值，测试集有缺失值则将该样本删除
         woe_dict.keys()
         del_index = []
         for key, value in woe_dict.items():
             if 'NA' not in value.keys() and 'NA' in df[key].unique():
                 index = np.where(df[key] == 'NA')
                 del_index.append(index)
             elif -7777 not in value.keys() and -7777 in df[key].unique():
                 index = np.where(df[key] == -7777)
                 del_index.append(index)
         ##删除样本
         if len(del_index) > 0:
             del_index = np.unique(del_index)
             df = df.drop(del_index)
             print('训练集无缺失值，但测试集有缺失值，该样本{0}删除'.format(del_index))
         
         ##WOE编码映射
         var_woe_name = []
         for key,value in woe_dict.items():
              val_name =  key+ "_woe"
              df[val_name] = df[key].map(value)
              var_woe_name.append(val_name)
         return df, var_woe_name

if __name__ == '__main__':
    path = 'D:\\code\\chapter5\\'
    data_path = os.path.join(path, 'data')
    file_name = 'german.csv'
    ##读取数据
    data_train, data_test = data_read(data_path,file_name)
    ##不可排序变量
    var_no_order = ['credit_history', 'purpose', 'personal_status', 'other_debtors',
                    'inst_plans', 'housing', 'job', 'telephone', 'foreign_worker']
    ##one-hot编码
    ##训练数据编码
    data_train.credit_history[882] = np.nan
    data_train_encode = onehot_encode(data_train[var_no_order], data_path, flag='train')
   
    ##测试集数据编码
    data_test.credit_history[529] = np.nan
    data_test.purpose[355] = np.nan
    data_test_encode = onehot_encode(data_test[var_no_order], data_path, flag='test')
    
    ##查看编码逆变化后的原始变量名
    df_encoded = data_test_encode.loc[0:4]
    data_inverse = onehot_encode(df_encoded, data_path, flag='transform')
    
    ##哑变量编码
    data_train_dummies = pd.get_dummies(data_train[var_no_order])
    data_test_dummies = pd.get_dummies(data_test[var_no_order])
    data_train_dummies.columns
    
    ##可排序变量
    ##注意，如果分类变量的标签为字符串，这是需要将字符串数值化才可以进行模型训练，标签编码其本质是为
    ##标签变量数值化而提出的方法，因此，其值支持单列数据的转化操作，并且转化后的结果是无序的。
    ##因此有序变量统一用字典映射的方式完成。
    var_order = ['status_account','svaing_account', 'present_emp', 'property']
    
    ##标签编码
    ##训练数据编码
    data_train_encode = label_encode(data_train[var_order[1]], data_path, flag='train')
    ##验证集数据编码
    data_test_encode = label_encode(data_test[var_order[1]], data_path, flag='test')
    ##查看编码变化后的原始变量名
    ##后面再改一下
    df_encoded = data_test_encode
    data_inverse = label_encode(df_encoded,data_path,flag='transform')
    
    ##自定义映射
    ##训练数据编码
    data_train.credit_history[882] = np.nan
    data_train_encode = dict_encode(data_train[var_order],data_path)
    ##测试集数据编码
    data_test.status_account[529] = np.nan
    data_test_encode = dict_encode(data_test[var_order],data_path)

    ##WOE编码
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values, var_woe_name = woe_encode(data_train, data_path, var_no_order, data_train.target, 'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = woe_encode(data_test, data_path, var_no_order, data_train.target, 'dict_woe_map', flag='test')




