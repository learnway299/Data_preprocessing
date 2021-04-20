#https://www.jianshu.com/p/c736027b85a1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option('max_columns',1000)
pd.set_option('max_row',300)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

#1数据探索分析
#返回缺失值个数以及最大、最小值
datafile = 'air_data.csv'

#读取原始数据，指定编码UTF-8
data = pd.read_csv(datafile, encoding='utf-8')
# print(data.head())

# 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）
explore = data.describe(percentiles = [], include = 'all').T# T是转置，转置后更方便查阅
#describe()函数自动计算非空值数，需要手动计算空值数
explore['null'] = len(data) - explore['count']
explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值']  #表头重命名
# print(explore)
# 导出结果
# explore.to_excel("result.xls")
# explore.to_markdown("result.md")

#2数据分布分析与汇总
#探索客户的基本信息分布情况
#客户信息类别
#提取会员入会年份
ffp = data['FFP_DATE'].apply(lambda x:datetime.strptime(x, '%Y/%m/%d'))
ffp_year = ffp.map(lambda x : x.year)

print(ffp_year)
#绘制各年份会员入会人数直方图
fig = plt.figure(figsize=(8, 5))  #设置画布大小
plt.hist(ffp_year, bins='auto', color='#0504aa')
plt.xlabel('年份')
plt.ylabel('入会人数')
plt.title('各年份会员入会人数')
plt.show()
plt.close()
# #提取会员不同性别人数
male = pd.value_counts(data['GENDER'])['男']
female = pd.value_counts(data['GENDER'])['女']
# #绘制会员性别比例饼图
fig = plt.figure(figsize=(7, 4))  #设置画布大小
plt.pie([male, female], labels=['男', '女'], colors=['lightskyblue', 'lightcoral'], autopct='%1.1f%%')
plt.title('会员性别比例')
plt.show()
plt.close()
#提取不同级别会员的人数
lv_four = pd.value_counts(data['FFP_TIER'])[4]
lv_five = pd.value_counts(data['FFP_TIER'])[5]
lv_six = pd.value_counts(data['FFP_TIER'])[6]
#绘制会员各级别人数条形图
# fig = plt.figure(figsize=(8,5))  #设置画布大小
plt.bar(range(3), [lv_four, lv_five, lv_six], width=0.4, alpha=0.8, color='skyblue')
#left：x轴的位置序列，一般采用arange函数产生一个序列；
#height：y轴的数值序列，也就是柱形图的高度，一般就是我们需要展示的数据；
#alpha：透明度
#width：为柱形图的宽度，一般这是为0.8即可；
#color或facecolor：柱形图填充的颜色；
plt.xticks([index for index in range(3)], ['4', '5', '6'])
plt.xlabel('会员等级')
plt.ylabel('会员人数')
plt.title('会员各级别人数')
plt.show()
plt.close()
#提取会员年龄
age = data['AGE'].dropna()
age = age.astype('int64')
#绘制会员年龄分布箱型图
fig = plt.figure(figsize=(5,10))
plt.boxplot(age, patch_artist=True, labels=['会员年龄'], boxprops={'facecolor': 'lightblue'}) #设置填充颜色
plt.title('会员年龄分布箱型图')
# 显示y坐标轴的底线
plt.grid(axis='y')
plt.show()
plt.close()
#.相关性分析
#提取属性并合并为新的数据集
data_corr = data.loc[:,['FFP_TIER', 'FLIGHT_COUNT', 'LAST_TO_END', 'SEG_KM_SUM', 'EXCHANGE_COUNT', 'Points_Sum']]

age1 = data['AGE'].fillna(0)
data_corr['AGE'] = age1.astype('int64')
data_corr['ffp_year'] = ffp_year

#计算相关性矩阵
dt_corr = data_corr.corr(method='pearson')
print('相关性矩阵为：\n', dt_corr)

#绘制热力图
plt.subplots(figsize=(10, 10))  #设置画面大小
# data:数据 square:是否是正方形 vmax:最大值 vmin:最小值 robust:排除极端值影响
sns.heatmap(dt_corr, annot=True, vmax=1, square=True, cmap='Blues')
plt.show()
plt.close()

#3 数据处理
print('原始数据的形状为：\n', data.shape)
#去除票价为空的记录
data_notnull = data.loc[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull(), :]
print('删除缺失记录后数据的形状为：\n', data_notnull.shape)
#只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录
index1 = data_notnull['SUM_YR_1'] != 0
index2 = data_notnull['SUM_YR_2'] != 0
index3 = (data_notnull['SEG_KM_SUM'] > 0) & (data_notnull['avg_discount'] != 0)
index4 = data_notnull['AGE'] > 100  #去除年龄大于100的记录
data_clean = data_notnull[(index1 | index2) & index3 & ~index4]
# data_clean.to_csv("data_cleaned.csv")
print('清洗后数据的形状为：\n', data_clean.shape)
print(data_clean)

#属性选择
#选取需求属性
data_selection = data_clean[['FFP_DATE', 'LOAD_TIME', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]
print('筛选的属性前5行为：\n', data_selection)
# #构造属性L
L = pd.to_datetime(data_selection['LOAD_TIME']) - pd.to_datetime(data_selection['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int')/30
# #合并属性
data_features = pd.concat([L, data_selection.iloc[:,2:]], axis=1)
print('构建的LRFMC属性前5行为：\n', data_features.head())
# #数据标准化
data_standar = StandardScaler().fit_transform(data_features)
# np.savez('airline_scale.npz', data_standar)#将标准化数据保存
print('标准化后LRFMC 5个属性为：\n', data_standar[:5,:])

#4 模型构建
# 读取标准化后的数据
# airline_scale = np.load('airline_scale.npz')['arr_0']
k = 5  #确定聚类中心数
#构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters=k, n_jobs=4, random_state=123)
fit_kmeans = kmeans_model.fit(data_standar)  #模型训练
# #查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_  #聚类中心
print('各类聚类中心为：\n', kmeans_cc)
kmeans_labels = kmeans_model.labels_  #样本的类别标签
print('各样本的类别标签为：\n', kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  #统计不同类别样本的数目
print('最终每个类别的数目为：\n', r1)
# #输出聚类分群的结果
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, columns=['ZL', 'ZR', 'ZF', 'ZM', 'ZC'])  #将聚类中心放在数据框中
print(cluster_center)
cluster_center.index = pd.DataFrame(kmeans_model.labels_).drop_duplicates().iloc[:,0]  #将样本类别作为数据框索引
print(cluster_center)
print(cluster_center.index)
print(pd.DataFrame(kmeans_model.labels_).drop_duplicates().iloc[:,0])

#客户价值分析
#客户分群雷达图
labels = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC']
legen = ['客户群' + str(i + 1) for i in cluster_center.index]
lstype = ['-', '--', (0, (3, 5, 1, 5, 1, 5)), ':', '-.']
kinds = list(cluster_center.iloc[:,0])
 #由于雷达图要保证数据闭合，因此要添加L列，并转换为np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])

 #分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
angle = np.concatenate((angle, [angle[0]]))

 #绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)  #以极坐标的形式绘制图形
#画线
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2, label=kinds[i])
#添加属性标签
ax.set_thetagrids(angle * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()
plt.close()