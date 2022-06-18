'''
time:2021/9/14
function:针对每个家庭，画出每个家庭的每个自变量与时间的变化因素
'''
import pandas as pd
import matplotlib.pyplot as plt
import os

house_data = pd.read_csv('house_data.csv', sep='\t', header=0)
name_list = ['datetime', 'eMeter', 'eMeterLow',
             'ePower', 'gasMeter', 'DD', 'FF', 'FX', 'N', 'P',
             'Q', 'T', 'T10', 'TD', 'U', 'VV', 'WW', 'gasPower']

# 住宅名称
# dwelling = house_data['dwelling'].unique()
# 'P01S01W0001'不存在
dwelling =['P01S01W7548', 'P01S02W0167', 'P01S01W5040' ,'P01S01W8669' ,'P01S01W0000',
 'P01S01W9617', 'P01S01W5588', 'P01S01W9431', 'P01S01W4002', 'P01S01W7042',
 'P01S01W6289', 'P01S01W5476', 'P01S02W4827', 'P01S01W1554' ,'P01S01W8828',
 'P01S01W2743', 'P01S01W5339', 'P01S01W1341', 'P01S01W5564', 'P01S01W2581',
 'P01S01W5746', 'P01S01W5292', 'P01S01W8655', 'P01S01W0998', 'P01S01W4979',
 'P01S01W6959', 'P01S01W4091', 'P01S02W6848', 'P01S01W0373', 'P01S02W4953',
 'P01S01W6595', 'P01S01W4489', 'P01S01W5855', 'P01S01W3955', 'P01S01W4313',
 'P01S01W6835', 'P01S01W8239', 'P01S01W1347', 'P01S02W7251', 'P01S01W8171',
 'P01S01W7980', 'P01S01W7071', 'P01S01W3497', 'P01S01W6495', 'P01S02W2995',
 'P01S01W0378', 'P01S01W4589', 'P01S01W4569', 'P01S01W6549',
 'P01S01W4579', 'P01S02W5065']

for i in range(len(dwelling)):
    name = dwelling[i]  # 读取用户住宅名称
    if not os.path.exists(name):
        os.mkdir(name)      # 创建文件夹存放图片
    name_file = name + '.csv'
    data = pd.read_csv(name_file, header=0, sep='\t')
    datatime = data['datetime']

    for j in range(len(name_list)-1):
        idx = data.iloc[:, j+1]
        # 画图
        plt.plot(datatime, idx)
        plt.title(name_list[j+1])
        path = name + '/' + name_list[j+1] + '.jpg'
        plt.savefig(path)
        plt.close()
    print("finish："+str(name))