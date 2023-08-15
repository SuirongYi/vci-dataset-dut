from scipy.stats import entropy
import numpy as np
import csv
from scipy.stats import norm
import pandas as pd
import os
from typing import Dict
from math import sqrt
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
ped_csv_file = 'data\\trajectories_filtered\\intersection_04_traj_ped_filtered.csv'
veh_csv_file = 'data\\trajectories_filtered\\intersection_04_traj_veh_filtered.csv'
ped_file_path = os.path.join(path, ped_csv_file)
veh_file_path = os.path.join(path, veh_csv_file)


def speed_data(data_path: str, ):
    tempt_list = list()
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tempt_list.append(sqrt(float(row['vx_est']) ** 2 + float(row['vy_est']) ** 2))
    # interval_width = (3.5 - 0.0) / 50
    # interval_bins = np.arange(0.0, 3.5 + interval_width, interval_width)
    # hist, bins = np.histogram(tempt_list, bins=interval_bins)
    # return hist / len(tempt_list)
    return tempt_list


def veh_distance_data(data_path_ped: str, data_path_veh):
    distance_list = []
    data_ = pd.read_csv(data_path_ped)
    colum_data = data_['frame'].tolist()
    frame_max = max(colum_data)
    temp_ped = {i: [] for i in range(1, frame_max + 1)}
    temp_veh = {i: [] for i in range(1, frame_max + 1)}
    with open(data_path_ped, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            temp_ped[int(row['frame'])].append([float(row['x_est']), float(row['y_est'])])
    with open(data_path_veh, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            temp_veh[int(row['frame'])].append([float(row['x_est']), float(row['y_est'])])
    for i in range(1, frame_max + 1):
        if temp_veh[i]:
            for v in temp_veh[i]:
                v_xy = np.array(v)
                for p in temp_ped[i]:
                    p_xy = np.array(p)
                    distance_list.append(np.linalg.norm(v_xy - p_xy))
    return distance_list


def ped_distance_data(data_path_ped: str):
    distance_list = []
    data_ = pd.read_csv(data_path_ped)
    colum_data = data_['frame'].tolist()
    frame_max = max(colum_data)
    temp_ped = {i: [] for i in range(1, frame_max + 1)}
    with open(data_path_ped, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            temp_ped[int(row['frame'])].append([float(row['x_est']), float(row['y_est'])])
    for i in range(1, frame_max+1):
        for p in temp_ped[i]:
            p_xy = np.array(p)
            for p_ in temp_ped[i]:
                if p != p_:
                    p_xy_ = np.array(p_)
                    distance = np.linalg.norm(p_xy-p_xy_)
                    if distance <= 0.7:
                        distance_list.append(0.0)
                    else:
                        distance_list.append(np.linalg.norm(p_xy-p_xy_)-0.7)
    return distance_list


def lateral_data(data_path_ped: str):
    ped_xy = dict()
    data_list = []
    with open(data_path_ped, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['id'] not in list(ped_xy.keys()):
                ped_xy[row['id']] = dict(x=[], y=[])
                ped_xy[row['id']]['x'].append(row['x_est'])
                ped_xy[row['id']]['y'].append(row['y_est'])
            else:
                ped_xy[row['id']]['x'].append(row['x_est'])
                ped_xy[row['id']]['y'].append(row['y_est'])
    for name, xy in ped_xy.items():
        star_xy = [float(xy['x'][0]), float(xy['y'][0])]
        end_xy = [float(xy['x'][-1]), float(xy['y'][-1])]
        range_ = len(xy['x'])
        for i in range(range_):
            distance = get_distance_from_point_to_line([float(xy['x'][i]), float(xy['x'][i])], star_xy, end_xy)
            data_list.append(distance)
    return data_list


def get_distance_from_point_to_line(point, line_point1, line_point2):
    a = line_point2[1] - line_point1[1]
    b = line_point1[0] - line_point2[0]
    c = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    distance = np.abs(a * point[0] + b * point[1] + c) / (np.sqrt(a ** 2 + b ** 2))
    return distance


# data = lateral_data(ped_file_path)
# data = veh_distance_data(ped_file_path, veh_file_path)
data = ped_distance_data(ped_file_path)
# data = speed_data(ped_file_path)
# # 拟合数据的分布
mu, std = norm.fit(data)

# 生成拟合后的分布曲线
x = np.linspace(min(data), max(data), 100)
y = norm.pdf(x, mu, std)

# 绘制原始数据直方图和拟合后的分布曲线
plt.hist(data, bins=200, density=True, alpha=0.6, color='r')
# plt.plot(x, y, 'r', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Pedestrian Distance Distribution')
plt.show()
