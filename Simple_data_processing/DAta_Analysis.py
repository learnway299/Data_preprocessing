import pandas as pd
import os
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Data_DIR = os.path.join(BASE_DIR, 'Data.csv')
df = pd.read_csv(Data_DIR, quoting=csv.QUOTE_NONE)
# 去除列名的双引号
col = []
for i in df.columns:
    col.append(i.strip('"')) # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符） 或字符序列。
df.columns = col
print(df.columns)
# 挑出时间列，特征1，特征2
df1 = df[['TIR_WRITETIME', 'TIR_VALUE1', 'TIR_VALUE2']]
print(df1.describe())
# 去除有无效值的行
df1 = df1.dropna()
# 处理特征1，特征2
# 去除特征1，特征2的双引号
df1['TIR_VALUE1'] = df1['TIR_VALUE1'].map(lambda x: x.strip('"'))
df1['TIR_VALUE2'] = df1['TIR_VALUE2'].map(lambda x: x.strip('"'))
# 字符串类型转换成数值型
df1['TIR_VALUE1'] = pd.to_numeric(df1['TIR_VALUE1'])
# to_numeric将参数转换为数字类型
df1['TIR_VALUE2'] = pd.to_numeric(df1['TIR_VALUE2'])
# 处理时间列
col1 = []
for i in df1['TIR_WRITETIME']:
    i = i[1:-1]
    col1.append(i.split(' ')[0] + ' ' + i.split(' ')[2])
print(col1)
# 转换成时间格式
df1['TIR_WRITETIME'] = col1
df1['lock_time'] = pd.to_datetime(df1['TIR_WRITETIME'])
# 重组数据
df1 = df1[['lock_time', 'TIR_VALUE1', 'TIR_VALUE2']]
print(df1.head())
# dt.date 和 dt.normalize()，他们都返回一个日期的 日期部分，即只包含年月日。
# 但不同的是date返回的Series是object类型的，normalize()返回的Series是datetime64类型的。
df_Ymd = pd.to_datetime(df1['lock_time']).dt.normalize()
df1['date'] = df_Ymd
# 按日期进行groupby
df2 = df1[['TIR_VALUE1', 'TIR_VALUE2']].groupby(df1['date']).agg('count')
print(df2.head())
print(df2.index)
# 了解数据分布
import matplotlib.pyplot as plt
df_2019 = df2['2019-01-01':'2019-12-30']
df_2020 = df2['2020-01-01':'2020-12-30']
df_2019['TIR_VALUE1'].plot.bar()
plt.show()
df_2020['TIR_VALUE1'].plot.bar()
plt.show()
print(df_2019.describe())
print(df_2020.describe())
#
print(df_2019.sort_values(by='TIR_VALUE1'))
print(df_2020.sort_values(by='TIR_VALUE1'))
#
col2 = []
for i in df_2019.index:
    col2.append(str(i))
print(col2)
df_2019.index = col2
df_2019_v1 = df_2019.sort_values(by='TIR_VALUE1')
df_2019_v1['TIR_VALUE1'].plot.bar()
plt.show()
col3 = []
for i in df_2020.index:
    col3.append(str(i))
print(col3)
df_2020.index = col3
df_2020_v1 = df_2020.sort_values(by='TIR_VALUE1')
df_2020_v1['TIR_VALUE1'].plot.bar()
plt.show()

Data_DIR1 = os.path.join(BASE_DIR, 'Data_1.csv')
df1.to_csv(Data_DIR1)
