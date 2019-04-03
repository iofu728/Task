# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-02-26 16:36:23
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-04 22:31:29
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import Series, DataFrame

from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['SimHei']
# rcParams['font.sans-serif'] = ['MicroSoft YaHei']

import pandas as pd


data_train = pd.read_csv('titanic/data/train.csv')
value_counts = data_train.Cabin.value_counts()
print(value_counts)

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(
    df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(
    df['Fare'].values.reshape(-1, 1), fare_scale_param)
df

# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数


plt.subplot2grid((2, 3), (0, 0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u"获救情况 (1为获救)")  # 标题
plt.ylabel(u"人数")

# plt.subplot2grid((2, 3), (0, 1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")

# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)")


# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")  # plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# # sets our legend for our graph.
# plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')


# plt.subplot2grid((2, 3), (1, 2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()
