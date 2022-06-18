'''
time:2021/9/14
function:读取原数据，清洗掉值为0的列特征,并将其切分成每个家庭的数据集
'''
import pandas as pd
import numpy as np

# 读取数据，依据某一列的值进行分组
house_data = pd.read_csv('house_data.csv', sep='\t', header=0)
name_list = ['datetime', 'eMeter', 'eMeterLow',
             'ePower', 'gasMeter', 'DD', 'FF', 'FX', 'N', 'P',
             'Q', 'T', 'T10', 'TD', 'U', 'VV', 'WW', 'gasPower']

# 删除具有缺失值的行，并替换掉原数据
# house_data.dropna(how='any', inplace=True)
# print(house_data)
# data = house_data['gasPower']
# print(data)
# print(np.mean(data))

house_data = pd.DataFrame(house_data)
house_data.drop(labels='eMeterReturn', axis=1, inplace=True)
house_data.drop(labels='eMeterLowReturn', axis=1, inplace=True)
house_data.drop(labels='ePowerReturn', axis=1, inplace=True)
house_data.drop(labels='DR', axis=1, inplace=True)
house_data.drop(labels='RG', axis=1, inplace=True)
house_data.drop(labels='SQ', axis=1, inplace=True)
print(house_data)
# # 用户的个数
dwelling = house_data['dwelling'].unique()
# print(dwelling)
# print(len(dwelling))  # 51个用户

# # 根据用户名切分出每个用户的数据集
for i in range(len(dwelling)):
    data = house_data.loc[house_data['dwelling']==dwelling[i]]
    name = dwelling[i] + '.csv'
    data = pd.DataFrame(data)
    # 删除倒数第二列-住宅名称
    data.drop('dwelling', axis=1, inplace=True)
    data.to_csv(name, sep='\t', header=name_list, index=None)

