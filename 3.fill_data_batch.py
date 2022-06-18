'''
author:郑书桦
time:2021/9/18
function: 使用随机森林回归（rfr） 和 最近邻方法（KNN）填补数据，并将填补后数据集存储为 '居民编号.csv' 文件
'''

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from pathlib import Path

# 随机森林回归方法填补
def fill_rfr(filename):
    name = os.path.split(filename) # 切割路径,将路径与文件名分开
    date_col_path = 'D:\\Study\\MTSDP\\data\\datetime\\' + name[1]
    outpath = 'D:\\Study\\MTSDP\\data\\rfr\\' + name[1] # 输出文件路径

    data = pd.read_csv(filename, delimiter='\t', parse_dates=['datetime'])
    # data.loc[:,['datetime']].to_csv(date_col_path) # 保存'datetime'列数据
    data.drop(data.columns[0], axis=1, inplace=True)  # 丢弃第一列索引列 --'datetime'

    sindex = np.argsort(data.isna().sum().values.tolist())  # 将有缺失值的列按缺失值的多少由小到大

    # 进入for循环进行空值填补
    for i in sindex:  # 按空值数量,从小到大进行排序来遍历
        if data.iloc[:, i].isna().sum() == 0:  # 将没有空值的行过滤掉
            continue  # 直接跳过当前的for循环

        df = data  # 复制df数据
        fillc = df.iloc[:, i]  # 将第i列的取出，之后作为y变量
        df = df.iloc[:, df.columns != df.columns[i]]  # 除了有这列以外的数据，之后作为X
        df_0 = SimpleImputer(missing_values=np.nan,  # 将df的数据全部用0填充
                             strategy="constant",
                             fill_value=0).fit_transform(df)
        Ytrain = fillc[fillc.notnull()]  # 在fillc列中,不为NAN的作为Y_train
        Ytest = fillc[fillc.isnull()]  # 在fillc列中,为NAN的作为Y_test
        Xtrain = df_0[Ytrain.index, :]  # 在df_0中(已经填充了0),中那些fillc列不为NAN的行作为Xtrain
        Xtest = df_0[Ytest.index, :]  # 在df_0中(已经填充了0),中那些fillc等于NAN的行作为X_test

        rfc = RandomForestRegressor()
        rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)  # Ytest为了定Xtest,以最后预测出Ypredict

        data.loc[data.iloc[:, i].isnull(), data.columns[i]] = Ypredict
        # 将data_copy中data_copy在第i列为空值的行,第i列,改成Ypredict

    # 数据整合
    datetime = pd.read_csv(date_col_path)
    # 添加'datetime'列
    data_new = datetime.merge(data, how='right', left_index=True, right_index=True)
    data_new.drop(data_new.columns[0], axis=1, inplace=True)  # 丢弃第一列

    data_new.to_csv(outpath, sep='\t')  # 另存csv文件

# 最近邻方法（KNN）填补
def fill_knn(filename):
    name = os.path.split(filename) # 切割路径,将路径与文件名分开
    date_col_path = 'D:\\Study\\MTSDP\\data\\datetime\\' + name[1]

    data = pd.read_csv(filename, delimiter='\t', parse_dates=['datetime'])
    # data.loc[:,['datetime']].to_csv(date_col_path) # 保存'datetime'列数据
    data.drop(data.columns[0], axis=1, inplace=True)  # 丢弃第一列索引列 --'datetime'

    feature_col = data.columns  # 原来dataframe的表头,即data_copy的表头

    # 使用knn
    imputer = KNNImputer(n_neighbors=10) # 取最近的10个数据的均值填补
    data = imputer.fit_transform(data)  # 此处data_copy的数据已经转换成numpy.ndarray格式
    data = pd.DataFrame(data, columns=feature_col)  # 将数据格式重新转换为dataframe

    # 数据整合
    datetime = pd.read_csv(date_col_path)
    data_new = datetime.merge(data, how='right', left_index=True, right_index=True)  # 添加'datetime'列
    data_new.drop(data_new.columns[0], axis=1, inplace=True)  # 丢弃第一列

    outpath = 'D:\\Study\\MTSDP\\data\\knn\\' + name[1] # 输出文件路径
    data_new.to_csv(outpath, sep='\t')  # 另存csv文件


p = Path(r"D:\Study\MTSDP\data\datasets")  # 初始化构造Path对象

FileList = list(p.glob("**/*.csv"))  # 得到所有的csv文件

for filename in FileList: # 遍历获取每个csv文件名
    # print(filename)
    # fill_rfr(filename)
    fill_knn(filename)
    break  # 测试一次就退出循环
