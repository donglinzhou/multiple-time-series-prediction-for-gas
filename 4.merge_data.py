'''
author:郑书桦
time:2021/9/19
function: 将同一目录下的csv文件合并并进行数据处理后输出
'''

import pandas as pd
from pathlib import Path

# 简单合并同一目录下的csv文件  --path:访问的文件夹路径; filename:合并后数据集输出路径
def merge_data(path, filename):
    p = Path(path) # 初始化构造Path对象
    # p_rfr = Path(r"D:\Study\MTSDP\data\rfr")
    # p_knn = Path(r"D:\Study\MTSDP\data\knn")

    p_list=  list(p.glob('*.csv')) # 查看同文件夹下的csv文件数

    print(u'共发现%s个CSV文件'% len(p_list))
    print(u'正在处理............')
    for i in p_list: # 循环读取同文件夹下的csv文件
        fr = open(i,'rb').read()
        with open(filename,'ab') as f: # 将结果保存为merged_result.csv
            f.write(fr)
    print(u'合并完毕！')

# 对合并的数据集进行数据处理
def mergerd_data_dp(filename_in, filename_out):
    # 删除重复表头 + 按时间戳排序
    df = pd.read_csv(filename_in, delimiter='\t', parse_dates=['datetime'], header=0)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop_duplicates(subset=None, keep=False, inplace=True) # 删除重复表头
    # print(df)
    df_sorted = df.sort_values(by=['datetime'], ignore_index=True) # 按'datetime'列进行升序排序
    # print(df_sorted)
    df_sorted = pd.DataFrame(df_sorted)

    dt_df = df_sorted.groupby(by='datetime').mean() # 以'datetime'列分组,并取平均
    dt_df.sort_index(inplace=True)
    dt_df.drop(dt_df.columns[0], axis=1, inplace=True)
    dt_df.to_csv(filename_out, sep='\t', header=True)

# 测试
file_path = 'D:\\Study\\MTSDP\\data\\knn' # 文件夹路径
filename = 'merged_result_knn.csv' # 简单合并后数据集路径
filename_final = 'merged_dataset_knn.csv' # 数据处理后数据集路径

merge_data(file_path, filename)
mergerd_data_dp(filename, filename_final)



