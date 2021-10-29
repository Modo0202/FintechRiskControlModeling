# -*- coding: utf-8 -*-
"""
第12章 样本不均衡处理
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler,NearMiss,TomekLinks,EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
from imblearn.ensemble import EasyEnsemble,BalanceCascade
import math
import warnings
warnings.filterwarnings("ignore") ##忽略警告
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

if __name__ == '__main__':
    
    path = 'D:\\code\\chapter12'
    data_path = os.path.join(path,'data')
    file_name = 'german.csv'
    ##读取数据
    data_train, data_test = data_read(data_path,file_name)
    data_df = pd.concat([data_train,data_test],axis=0)
    data_df = data_df.reset_index(drop=True)
    sum( data_df.target == 1 )
    sum( data_df.target == 0 )
    x_data = data_df.loc[:,data_df.columns != 'target']
    y_data = data_df.target
   
    #############数据层下采样方法###################################################
    ##随机下采样
    rand_under_sample= RandomUnderSampler(random_state=10, replacement=True,ratio=1)
    X_resample, y_resample = rand_under_sample.fit_sample(x_data, y_data)
    X_res = pd.DataFrame(X_resample,columns=x_data.columns)
    X_res.shape
    ##查看下采样索引
    rand_under_sample.sample_indices_
    ##PCA方法降维
    cont_name = ['duration', 'amount', 'income_rate',  'residence_info',  
               'age',  'num_credits','dependents']
    x_cont_data = data_df[cont_name]
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_res[cont_name])

    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_data[rand_under_sample.sample_indices_]==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('RandomUnderSampler',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    

    ##nearmiss方法
    near_miss_1 = NearMiss(random_state=10, version=1,n_neighbors=3,ratio=1)
    X_resample_1, y_resample_1 = near_miss_1.fit_sample(x_cont_data, y_data)
    near_miss_2 = NearMiss(random_state=10, version=2,n_neighbors=3,ratio=1)
    X_resample_2, y_resample_2 = near_miss_2.fit_sample(x_cont_data, y_data)
    near_miss_3 = NearMiss(random_state=10, version=3,n_neighbors=3,ratio=1)
    X_resample_3, y_resample_3 = near_miss_3.fit_sample(x_cont_data, y_data)
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample_1)
    X_pca_2 = model_pca.transform(X_resample_2)
    X_pca_3 = model_pca.transform(X_resample_3)
    
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(221)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_1降维结果
    plt.subplot(222)
    index_2 = y_resample_1==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('NearMiss_1',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_2降维结果
    plt.subplot(223)
    index_3 = y_resample_2==0
    plt.scatter(X_pca_2[index_3, 0], X_pca_2[index_3, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_2[~index_3, 0], X_pca_2[~index_3, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('NearMiss_2',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_3降维结果
    plt.subplot(224)
    index_4 = y_resample_3==0
    plt.scatter(X_pca_3[index_4, 0], X_pca_3[index_4, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_3[~index_4, 0], X_pca_3[~index_4, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('NearMiss_3',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    
    
    ##TomekLinks方法
    tom_link = TomekLinks(sampling_strategy='auto',random_state=10)
    X_resample, y_resample = tom_link.fit_sample(x_cont_data, y_data)
    tom_link.sample_indices_
    
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample)
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_resample==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('TomekLinks Sample',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    
 
    ##EditedNearestNeighbours方法
    Enn = EditedNearestNeighbours(random_state=10,n_neighbors=3,kind_sel='mode')
    X_resample, y_resample = Enn.fit_sample(x_cont_data, y_data)
    Enn.sample_indices_
    
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample)
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_resample==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('EditedNearestNeighbours',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()

    ##bagging方法
    sub_num = math.ceil(sum( data_df.target == 0 ) / sum( data_df.target == 1 ))
    easy_en = EasyEnsemble(random_state=10, n_subsets=sub_num,replacement=True)
    X_resample, y_resample = easy_en.fit_sample(x_cont_data, y_data)
    X_resample.shape
    y_resample.shape

    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample[0,:,:])
    X_pca_2 = model_pca.transform(X_resample[1,:,:])
    X_pca_3 = model_pca.transform(X_resample[2,:,:])
    
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(221)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##sample_1降维结果
    plt.subplot(222)
    index_2 =  y_resample[0,:]==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('sample_1',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##sample_2降维结果
    plt.subplot(223)
    index_3 =  y_resample[1,:]==0
    plt.scatter(X_pca_2[index_3, 0], X_pca_2[index_3, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_2[~index_3, 0], X_pca_2[~index_3, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('sample_2',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##Nsample_3降维结果
    plt.subplot(224)
    index_4 =  y_resample[2,:]==0
    plt.scatter(X_pca_3[index_4, 0], X_pca_3[index_4, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_3[~index_4, 0], X_pca_3[~index_4, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('sample_3',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    
    ##boosting方法
#    boost_balance = BalanceCascade(random_state=10,estimator=DecisionTreeClassifier())
#    X_resample, y_resample = boost_balance.fit_resample(x_cont_data, y_data)
#    X_resample.shape
#    y_resample.shape
#    
#    ##PCA降维，到2维
#    model_pca = PCA(n_components=2).fit(x_cont_data)
#    X_pca_raw = model_pca.transform(x_cont_data)
#    X_pca_1 = model_pca.transform(X_resample[0,:,:])
#    X_pca_2 = model_pca.transform(X_resample[1,:,:])
#    X_pca_3 = model_pca.transform(X_resample[2,:,:])
#    
#    ##降维后,用前两维的
#    plt.figure(figsize=(15, 8))
#    fontsize_1 = 15
#    ##原始数据降维结果
#    plt.subplot(221)
#    index_1 = y_data==0
#    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
#    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
#    plt.title('raw_data',fontsize=fontsize_1)
#    plt.xticks( fontsize=fontsize_1)
#    plt.yticks( fontsize=fontsize_1)
#    plt.legend()
#    ##NearMiss_1降维结果
#    plt.subplot(222)
#    index_2 =  y_resample[0,:]==0
#    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
#    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
#    plt.title('sample_1',fontsize=fontsize_1)
#    plt.xticks( fontsize=fontsize_1)
#    plt.yticks( fontsize=fontsize_1)
#    plt.legend()
#    ##NearMiss_2降维结果
#    plt.subplot(223)
#    index_3 =  y_resample[1,:]==0
#    plt.scatter(X_pca_2[index_3, 0], X_pca_2[index_3, 1], c='grey',marker='o',label='负样本')
#    plt.scatter(X_pca_2[~index_3, 0], X_pca_2[~index_3, 1], c='black',marker='+',alpha=0.5,label='正样本')
#    plt.title('sample_2',fontsize=fontsize_1)
#    plt.xticks( fontsize=fontsize_1)
#    plt.yticks( fontsize=fontsize_1)
#    plt.legend()
#    ##NearMiss_3降维结果
#    plt.subplot(224)
#    index_4 =  y_resample[2,:]==0
#    plt.scatter(X_pca_3[index_4, 0], X_pca_3[index_4, 1], c='grey',marker='o',label='负样本')
#    plt.scatter(X_pca_3[~index_4, 0], X_pca_3[~index_4, 1], c='black',marker='+',alpha=0.5,label='正样本')
#    plt.title('sample_3',fontsize=fontsize_1)
#    plt.xticks( fontsize=fontsize_1)
#    plt.yticks( fontsize=fontsize_1)
#    plt.legend()
    
    #############数据层上采样方法###################################################
    ##随机上采样
    rand_over_sample= RandomOverSampler(random_state=10,sampling_strategy=1)
    X_resample, y_resample = rand_over_sample.fit_sample(x_data, y_data)
    X_res = pd.DataFrame(X_resample,columns=x_data.columns)
    X_resample.shape
  
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_res[cont_name])

    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_resample==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('RandomOverSampler',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()

    ##Smote样本生成方法
    sm_sample = SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5,kind='regular')
    X_resample, y_resample = sm_sample.fit_resample(x_cont_data, y_data)
    X_resample.shape
    
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample)

    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_resample==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('Smote',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    
    
    ##Borderline SMOTE-1样本生成
    sm_sample_1 = SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5,kind='borderline1')
    X_resample_1, y_resample_1 = sm_sample_1.fit_resample(x_cont_data, y_data)
    X_resample_1.shape
    ##Borderline SMOTE-2样本生成
    sm_sample_2 = SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5,kind='borderline2')
    X_resample_2, y_resample_2 = sm_sample_2.fit_resample(x_cont_data, y_data)

    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample)
    X_pca_2 = model_pca.transform(X_resample_1)
    X_pca_3 = model_pca.transform(X_resample_2)
    
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(221)
    index_1 = y_data==0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('raw_data',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(222)
    index_2 = y_resample==0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('Smote',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##Borderline SMOTE-1降维结果
    plt.subplot(223)
    index_3 =  y_resample_1==0
    plt.scatter(X_pca_2[index_3, 0], X_pca_2[index_3, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_2[~index_3, 0], X_pca_2[~index_3, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('Borderline SMOTE-1',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()
    ##Borderline SMOTE-2降维结果
    plt.subplot(224)
    index_4 =  y_resample_2==0
    plt.scatter(X_pca_3[index_4, 0], X_pca_3[index_4, 1], c='grey',marker='o',label='负样本')
    plt.scatter(X_pca_3[~index_4, 0], X_pca_3[~index_4, 1], c='black',marker='+',alpha=0.5,label='正样本')
    plt.title('Borderline SMOTE-2',fontsize=fontsize_1)
    plt.xticks( fontsize=fontsize_1)
    plt.yticks( fontsize=fontsize_1)
    plt.legend()

    






  
    