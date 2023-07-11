from typing import Dict, Tuple, List
import numpy as np
from dataclasses import dataclass
from math import sin, cos, acos, sqrt, pi
import matplotlib.patches as patches
from numpy import ndarray
from DataClass import TrafficVeh, SurrData
from Model import TrafficParticipant, Model
import matplotlib.pyplot as plt
import copy
import time
import csv
import os
import pandas as pd

width = 1.6
length = 3.8


@dataclass
class SurrInit:
    # setting: Setting
    ped_data: Dict
    traffic_vehs: Dict[str, TrafficVeh]
    para_list: List


@dataclass
class SurrUpdate:
    traffic_vehs: Dict[str, TrafficVeh]


@dataclass
class SurrResult:
    surr: SurrData


class Surrounding:
    def __init__(self, init: SurrInit):
        # self.init = init
        self.ped_data = init.ped_data
        self.para_data = init.para_list
        self.traffic_info: Dict[str, TrafficVeh] = dict()
        self.ego_veh = self.get_object(init.traffic_vehs)
        self.ego_data: Dict[str, TrafficParticipant] = dict()
        self.ego_list = list(self.ego_veh.keys())
        for name in self.ego_list:
            self.ego_data[name] = TrafficParticipant(**self.ego_veh[name], id=name)  # 被控对象初始化
        self.model = Model(tau=0.03)
        self.all_object_result: SurrData = SurrData()

    def update(self, update: SurrUpdate) -> SurrResult:
        self.reset()
        for name, veh in update.traffic_vehs.items():
            if veh.type in ('pedestrian', 'bicycle') and name not in self.ego_list:
                self.add_participant(veh)
        self.traffic_info = copy.deepcopy(update.traffic_vehs)
        for obj in self.ego_data.values():
            self.update_sur_veh(obj)
            update_target_point(obj)
        self.del_participant()
        self.model.control_object = self.ego_data
        self.all_object_result.ped = self.model.update()

        return SurrResult(surr=self.all_object_result)

    def update_sur_veh(self, obj: TrafficParticipant):
        for veh in self.traffic_info.values():
            if veh.id == obj.id: continue
            if veh.type[0] == 'v':
                head_point, tail_point = get_veh_point(veh)
                if select(obj, head_point) or select(obj, tail_point):
                    obj.sur_vehicle.append(veh)
            else:
                if select(obj, np.array([veh.x, veh.y])):
                    obj.sur_ped_bic.append(veh)

    def reset(self):
        self.all_object_result.reset()

    def get_object(self, traffic: Dict[str, TrafficVeh]) -> Dict[str, Dict]:
        control_object = dict()
        for name, tp in traffic.items():
            if tp.id[0] == 'v':
                continue
            control_object[name] = dict(type=tp.type, original_x=tp.x, original_y=tp.y, u=tp.u,
                                        phi=tp.phi, target_points=np.array([self.ped_data[name]['end_xy']]),
                                        adjust_time=0.95, view_radius=10.0, view_angle=2 * pi,
                                        A_sur=self.para_data[0], B_sur=self.para_data[1],
                                        A_veh=self.para_data[2], B_veh=self.para_data[3],
                                        r_x_max=4.50, pre_horizon=0.6
                                        )
        return control_object

    def add_participant(self, obj: TrafficVeh):
        self.ego_list.append(obj.id)
        self.ego_data[obj.id] = TrafficParticipant(id=obj.id, type=obj.type, original_x=obj.x, original_y=obj.y,
                                                   u=obj.u, phi=obj.phi,
                                                   target_points=np.array([self.ped_data[obj.id]['end_xy']]),
                                                   adjust_time=0.95, view_radius=10.0, view_angle=2 * pi,
                                                   A_sur=self.para_data[0], B_sur=self.para_data[1],
                                                   A_veh=self.para_data[2], B_veh=self.para_data[3],
                                                   r_x_max=4.50, pre_horizon=0.6
                                                   )

    def del_participant(self):
        del_list = []
        for name, obj in self.ego_data.items():
            distance = np.linalg.norm(obj.target_point[:2] - np.array([obj.x, obj.y]))
            if distance <= 0.1:
                del_list.append(name)
        self.ego_list = [i for i in self.ego_list if i not in del_list]
        self.ego_data = {k: v for k, v in self.ego_data.items() if k in self.ego_list}


def update_target_point(obj: TrafficParticipant):
    ego_position = np.array([obj.x, obj.y])
    d = np.linalg.norm(ego_position - obj.target_point)
    if d <= 3.0:
        if obj.curr_index < obj.size - 1:
            # print('更新目标点')
            obj.reset_original_point(obj.curr_index + 1)


def get_veh_point(vehicle: TrafficVeh) -> Tuple:
    head_x = vehicle.x + vehicle.length / 2 * cos(vehicle.phi)
    head_y = vehicle.y + vehicle.length / 2 * sin(vehicle.phi)
    tail_x = vehicle.x - vehicle.length / 2 * cos(vehicle.phi)
    tail_y = vehicle.y - vehicle.length / 2 * sin(vehicle.phi)
    return np.array([head_x, head_y]), np.array([tail_x, tail_y])


def select(obj: TrafficParticipant, point: ndarray) -> bool:
    ego_position = np.array([obj.x, obj.y])
    d_vector = point - ego_position
    distances = np.linalg.norm(d_vector)
    ego_v = np.array([cos(obj.phi), sin(obj.phi)])
    theta = acos(np.clip(ego_v.dot(d_vector) / distances,-1, 1))
    if distances <= obj.view_radius and theta <= obj.view_angle / 2:
        return True
    else:
        return False


def get_ped_data(data_path: str, ) -> Tuple[Dict, Dict, Dict]:
    data = pd.read_csv(data_path)
    colum_data = data['frame'].tolist()
    frame_max = max(colum_data)
    ped_list = {i: [] for i in range(1, frame_max + 1)}
    ped_xy = dict()
    pedestrian = dict()
    test_ped_data = dict()
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if 'ped' + row['id'] not in test_ped_data.keys():
                test_ped_data['ped' + row['id']] = []
                test_ped_data['ped' + row['id']].append(int(row['frame']))
            else:
                test_ped_data['ped' + row['id']].append(int(row['frame']))
            ped_list[int(row['frame'])].append('ped' + row['id'])
            if 'ped' + row['id'] not in list(ped_xy.keys()):
                ped_xy['ped' + row['id']] = dict(x=[], y=[], u=[], phi=[])
                ped_xy['ped' + row['id']]['x'].append(row['x_est'])
                ped_xy['ped' + row['id']]['y'].append(row['y_est'])
                ped_xy['ped' + row['id']]['u'].append(sqrt(float(row['vx_est']) ** 2 + float(row['vy_est']) ** 2))
                ped_xy['ped' + row['id']]['phi'].append(np.arctan2(float(row['vy_est']), float(row['vx_est'])))
            else:
                ped_xy['ped' + row['id']]['x'].append(row['x_est'])
                ped_xy['ped' + row['id']]['y'].append(row['y_est'])
                ped_xy['ped' + row['id']]['u'].append(sqrt(float(row['vx_est']) ** 2 + float(row['vy_est']) ** 2))
                ped_xy['ped' + row['id']]['phi'].append(np.arctan2(float(row['vy_est']), float(row['vx_est'])))
    for name, xy in ped_xy.items():
        pedestrian[name] = dict(star_xy=[float(xy['x'][0]), float(xy['y'][0])],
                                end_xy=[float(xy['x'][-1]), float(xy['y'][-1]), 1.80],
                                star_u=xy['u'][0],
                                star_phi=xy['phi'][0])
    ped_frame = dict()
    for name, frame in test_ped_data.items():
        if frame[0] not in ped_frame.keys():
            ped_frame[frame[0]] = []
            ped_frame[frame[0]].append(name)
        else:
            ped_frame[frame[0]].append(name)
    return ped_list, pedestrian, ped_frame


def get_veh_data(data_path: str) -> Dict:
    data = pd.read_csv(data_path)
    colum_data = data['frame'].tolist()
    frame_max = max(colum_data)
    veh_dict = {i: {} for i in range(1, frame_max + 1)}
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            veh_dict[int(row['frame'])]['veh' + row['id']] = [float(row['x_est']), float(row['y_est']),
                                                              float(row['psi_est']), float(row['vel_est'])]
    return veh_dict


def plot_vehicle(x, y, phi):
    x_a = x + width / 2 * sin(phi) - length / 2 * cos(phi)
    y_a = y - width / 2 * cos(phi) - length / 2 * sin(phi)
    x_b = x_a + length * cos(phi)
    y_b = y_a + length * sin(phi)
    x_c = x_a - width * sin(phi)
    y_c = y_a + width * cos(phi)
    x_d = x_a + length * cos(phi) - width * sin(phi)
    y_d = y_a + length * sin(phi) + width * cos(phi)
    ax.plot([x_a, x_b], [y_a, y_b], color='blue', zorder=3)
    ax.plot([x_a, x_c], [y_a, y_c], color='blue', zorder=3)
    ax.plot([x_d, x_b], [y_d, y_b], color='blue', zorder=3)
    ax.plot([x_d, x_c], [y_d, y_c], color='blue', zorder=3)


def plot_veh_data(data_path: str, ) -> Dict:
    data = pd.read_csv(data_path)
    colum_data = data['frame'].tolist()
    frame_max = max(colum_data)
    temp = {i: [] for i in range(1, frame_max + 1)}
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['label'] == 'ped':
                temp[int(row['frame'])].append([float(row['x_est']), float(row['y_est'])])
            else:
                temp[int(row['frame'])].append([float(row['x_est']), float(row['y_est']), float(row['psi_est'])])
    return temp


def real_ped_data(data_path: str, ) -> np.ndarray:
    tempt_list = list()
    with open(data_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tempt_list.append(sqrt(float(row['vx_est']) ** 2 + float(row['vy_est']) ** 2))
    interval_width = (3.5 - 0.0) / 50
    interval_bins = np.arange(0.0, 3.5 + interval_width, interval_width)
    hist, bins = np.histogram(tempt_list, bins=interval_bins)
    return hist / len(tempt_list)


def simulation_data(ped_path: str, veh_path: str, para_list: List) -> np.ndarray:
    ped_list, ped_data, test = get_ped_data(ped_path)
    veh_data = get_veh_data(veh_path)
    simulation_step = len(ped_list)
    traffic_vehs = dict()
    sur_init = SurrInit(ped_data=ped_data, traffic_vehs=traffic_vehs, para_list=para_list)
    sur = Surrounding(init=sur_init)
    v_data = []
    plot_ped_data = {i: [] for i in range(1, simulation_step + 1)}
    for i in range(1, simulation_step + 1):
        veh_frame = len(veh_data)
        if i < veh_frame + 1:
            for n, v in veh_data[i].items():
                traffic_vehs[n] = TrafficVeh(id=n, type='vehicle', x=v[0], y=v[1], phi=v[2],
                                             u=v[3], length=3.8, width=1.6)
        for n, p in sur.ego_data.items():
            traffic_vehs[n] = TrafficVeh(id=n, type='pedestrian', x=p.x, y=p.y, phi=p.phi,
                                         u=p.u)
        if i in list(test.keys()):
            for p in test[i]:
                if p not in sur.ego_list:
                    traffic_vehs[p] = TrafficVeh(id=p, type='pedestrian', x=ped_data[p]['star_xy'][0],
                                                 y=ped_data[p]['star_xy'][1], phi=ped_data[p]['star_phi'],
                                                 u=ped_data[p]['star_u'])
        sur_list = sur.update(update=SurrUpdate(traffic_vehs=traffic_vehs))
        for value in sur.ego_data.values():
            plot_ped_data[i].append([value.x, value.y])
            v_data.append(value.u)
        for ped in sur_list.surr.ped:
            sur.ego_data[ped.id].update(x=ped.x, y=ped.y, u=ped.u, phi=ped.phi)
            sur.ego_data[ped.id].set_v()
            sur.ego_data[ped.id].set_coordinate()
            sur.ego_data[ped.id].history_x.append(ped.x)
            sur.ego_data[ped.id].history_y.append(ped.y)
            sur.ego_data[ped.id].history_u.append(ped.u)
            sur.ego_data[ped.id].history_phi.append(ped.phi)
        traffic_vehs.clear()
    interval_width = (3.5 - 0.0) / 50
    interval_bins = np.arange(0.0, 3.5 + interval_width, interval_width)
    hist, bins = np.histogram(v_data, bins=interval_bins)
    return hist/len(v_data)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)
    ped_csv_file = 'data\\trajectories_filtered\\intersection_04_traj_ped_filtered.csv'
    veh_csv_file = 'data\\trajectories_filtered\\intersection_04_traj_veh_filtered.csv'
    ped_file_path = os.path.join(path, ped_csv_file)
    veh_file_path = os.path.join(path, veh_csv_file)

    # data = simulation_data(ped_file_path, veh_file_path, [0.68, 0.07, 4.4, 3.09])
    # print(data)
    # [0.85, 1.95, 3.50, 0.40]
    ped_list, ped_data, test = get_ped_data(ped_file_path)
    veh_data = get_veh_data(veh_file_path)
    simulation_step = len(ped_list)
    traffic_vehs = dict()
    sur_init = SurrInit(ped_data=ped_data, traffic_vehs=traffic_vehs, para_list=[0.85, 1.95, 3.50, 0.40])
    sur = Surrounding(init=sur_init)
    t = list()
    v_data = []
    if_plot = True

    plot_ped_data = {i: [] for i in range(1, simulation_step + 1)}
    for i in range(1, simulation_step + 1):
        veh_frame = len(veh_data)
        if i < veh_frame + 1:
            for n, v in veh_data[i].items():
                traffic_vehs[n] = TrafficVeh(id=n, type='vehicle', x=v[0], y=v[1], phi=v[2],
                                             u=v[3], length=3.8, width=1.6)
        for n, p in sur.ego_data.items():
            traffic_vehs[n] = TrafficVeh(id=n, type='pedestrian', x=p.x, y=p.y, phi=p.phi,
                                         u=p.u)
        if i in list(test.keys()):
            for p in test[i]:
                if p not in sur.ego_list:
                    traffic_vehs[p] = TrafficVeh(id=p, type='pedestrian', x=ped_data[p]['star_xy'][0],
                                                 y=ped_data[p]['star_xy'][1], phi=ped_data[p]['star_phi'],
                                                 u=ped_data[p]['star_u'])
        t1 = time.time()
        sur_list = sur.update(update=SurrUpdate(traffic_vehs=traffic_vehs))
        for value in sur.ego_data.values():
            plot_ped_data[i].append([value.x, value.y])
            v_data.append(value.u)
        for ped in sur_list.surr.ped:
            sur.ego_data[ped.id].update(x=ped.x, y=ped.y, u=ped.u, phi=ped.phi)
            sur.ego_data[ped.id].set_v()
            sur.ego_data[ped.id].set_coordinate()
            sur.ego_data[ped.id].history_x.append(ped.x)
            sur.ego_data[ped.id].history_y.append(ped.y)
            sur.ego_data[ped.id].history_u.append(ped.u)
            sur.ego_data[ped.id].history_phi.append(ped.phi)
        traffic_vehs.clear()
        t2 = time.time()
        t.append(t2 - t1)
    # print(np.mean(np.array(t)))
    # print(v_data)

    # plot = PlotDate(traffic_dict=traffic_vehs, control_obj=sur.ego_data, step=simulation_step)
    # dp = DynamicPlot(plot)
    # ped0 = sur.ego_data['bic0']
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # axes[0, 0].plot(ped0.history_a)
    # axes[0, 0].set_title('a')
    # axes[0, 1].plot(ped0.history_x, ped0.history_y)
    # axes[0, 1].axis('equal')
    # axes[0, 1].set_title('position')
    # axes[1, 0].plot(ped0.history_u)
    # axes[1, 0].set_title('u')
    # axes[1, 1].plot(ped0.history_phi)
    # axes[1, 1].set_title('phi')
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.axis('equal')
    # ax.set_title('trajectory')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # for tp in sur.ego_data.values():
    #     ax.plot(tp.history_x, tp.history_y, 'r')
    # plt.show()
    # print(plot_ped_data)

    if if_plot:
        plot_veh_data = plot_veh_data(veh_file_path)
        fig = plt.figure(figsize=(10, 8))
        plt.ion()
        ax = fig.add_subplot()
        for i in range(1, len(plot_ped_data) + 1):
            plt.cla()
            for ped in plot_ped_data[i]:
                circle = patches.Circle(xy=(ped[0], ped[1]), radius=0.15, fc='red', ec='red', zorder=3)
                ax.add_patch(circle)
            if i in list(plot_veh_data.keys()):
                for veh in plot_veh_data.get(i):
                    plot_vehicle(veh[0], veh[1], veh[2])
            ax.axis('equal')
            ax.set_xlim(-5.0, 35.0)
            ax.set_ylim(-5.0, 35.0)
            plt.draw()
            plt.pause(0.03)
        plt.show()
        plt.ioff()
        plt.clf()
