import numpy as np
import csv
import os
from typing import List
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from fitter import Fitter

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
ped_csv_file = 'data\\trajectories_filtered\\intersection_04_traj_ped_filtered.csv'
ped_file_path = os.path.join(path, ped_csv_file)


def get_ped_data(data_path: str, ):
    # csv_file = os.path.join(path, data_path)
    tempt_list = list()
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tempt_list.append(math.sqrt(float(row['vx_est'])**2 + float(row['vy_est'])**2))
    # interval_width = (4.0 - 0.0) / 100
    # interval_bins = np.arange(0.0, 4.0 + interval_width, interval_width)
    # hist, bins = np.histogram(tempt_list, bins=interval_bins)
    return tempt_list


ped_data = get_ped_data(ped_file_path)
# interval_width = (4.0 - 0.0) / 100
#
# # 计算每个小区间的边界
# interval_bins = np.arange(0.0, 4.0 + interval_width, interval_width)
#
# # 统计数据在每个小区间中的频率
# hist, bins = np.histogram(ped_data, bins=100, range=(min(ped_data), max(ped_data)))
# print(ped_data)

mu, std = norm.fit(ped_data)
#
x = np.linspace(min(ped_data), max(ped_data), 100)
y = norm.pdf(x, mu, std)

plt.hist(ped_data, bins=50, density=True, alpha=0.6, color='g')
plt.plot(x, y, 'r', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Fitted Distribution')
plt.show()


# 创建 Fitter 对象并拟合数据
# fitter = Fitter(ped_data)
# fitter.fit()
#
# # 获取拟合结果
# fitter.summary()
# # 获取拟合效果最佳的分布
# # 获取拟合效果最佳的分布
# best_fit = fitter.get_best()
#
# # 绘制拟合曲线
# fitter.plot_pdf(names=None, Nbest=3, lw=2)
# plt.hist(ped_data, bins=30, density=True, alpha=0.6, color='g')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Fitted Distribution')
# plt.show()


