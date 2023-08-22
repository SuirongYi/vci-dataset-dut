from typing import Dict, Tuple, List
import numpy as np
from dataclasses import dataclass
from math import sin, cos, acos, sqrt, pi
import matplotlib.patches as patches
from numpy import ndarray
from DataClass import TrafficVeh, SurrData, SurrInfo
from Model import TrafficParticipant, Model
import matplotlib.pyplot as plt
import copy
import time
import random

width = 1.5
length = 3.5


@dataclass
class SurrInit:
    ped_data: Dict   # 初始化行人位置参数
    para_list: List  # 社会力参数列表


@dataclass
class SurrUpdate:
    traffic_vehs: Dict[str, TrafficVeh]


@dataclass
class SurrResult:
    surr: SurrData


class Surrounding:
    def __init__(self, init: SurrInit):
        self.para_data = init.para_list
        self.traffic_info: Dict[str, TrafficVeh] = dict()
        self.ego_veh: Dict[str, Dict] = init.ped_data
        self.ego_data: Dict[str, TrafficParticipant] = dict()  # 被控行人字典
        self.ego_list = list(self.ego_veh.keys())
        for name in self.ego_list:
            self.ego_data[name] = TrafficParticipant(**self.ego_veh[name], id=name,adjust_time=0.40, 
                                                     view_radius=11.0, view_angle=2 * pi / 3,
                                                     A_sur=self.para_data[0], B_sur=self.para_data[1],
                                                     A_veh=self.para_data[2], B_veh=self.para_data[3],
                                                     r_x_max=100, pre_horizon=1.5)                    # 被控对象初始化
        self.model = Model(tau=0.1)
        self.all_object_result: SurrData = SurrData()

    def update(self, update: SurrUpdate) -> SurrResult:
        self.reset()
        for name, veh in update.traffic_vehs.items():
            if veh.type in ('pedestrian', 'bicycle') and name not in self.ego_list:
                self.add_participant(veh)
        self.traffic_info = copy.deepcopy(update.traffic_vehs)
        for obj in self.ego_data.values():
            obj.reset()
            self.update_sur_veh(obj)    # 更新周围交通参与者
            update_target_point(obj)    #todo 决策过程，待修改
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

    def add_participant(self, obj: TrafficVeh):
        self.ego_list.append(obj.id)
        self.ego_data[obj.id] = TrafficParticipant(id=obj.id, type=obj.type, original_x=obj.x, original_y=obj.y,
                                                   u=obj.u, phi=obj.phi,
                                                   adjust_time=0.40, view_radius=11.0, view_angle=2 * pi / 3,
                                                   A_sur=self.para_data[0], B_sur=self.para_data[1],
                                                   A_veh=self.para_data[2], B_veh=self.para_data[3],
                                                   r_x_max=10, pre_horizon=1.5)

    def del_participant(self, obj: SurrInfo):
        distance = np.linalg.norm(self.ego_data[obj.id].target_point[:2] - np.array([obj.x, obj.y]))
        if distance <= 0.8:
            self.ego_list = [i for i in self.ego_list if i != obj.id]
            self.ego_data = {k: v for k, v in self.ego_data.items() if k in self.ego_list}


def update_target_point(obj: TrafficParticipant):
    ego_position = np.array([obj.x, obj.y])
    d = np.linalg.norm(ego_position - obj.target_point)
    if d <= 1.0:
        if obj.curr_index < obj.size - 1:
            # print('更新目标点')
            # probability = random.random()
            # if probability <= 0.2:
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
    theta = acos(np.clip(ego_v.dot(d_vector) / distances, -1, 1))
    if distances <= obj.view_radius and theta <= obj.view_angle / 2:
        return True
    else:
        return False


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


if __name__ == "__main__":
    ped_data = dict(ped0=dict(original_x=15, original_y=3.0, phi=0.0, u=0.0,
                              target_points=np.array([[8.0, 4.0, 1.70, 1]])),
                    ped1=dict(original_x=1.0, original_y=3.0, phi=0.0, u=0.0,
                              target_points=np.array([[7.0, 4.0, 1.70, 1],
                                                      [7.5, 25.0, 1.80, 0]])),
                    )
                    # ped1=dict(x=0.0, y=0.0, phi=0.0, u=0.0))
    simulation_step = 150
    traffic_vehs = dict()
    sur_init = SurrInit(ped_data=ped_data, para_list=[0.85, 1.95, 3.50, 0.40])
    sur = Surrounding(init=sur_init)
    t = list()
    if_plot = True
    plot_ped_data = {i: [] for i in range(simulation_step+1)}
    for i in range(0, simulation_step + 1):
        # print(f'第{i}/{simulation_step + 1}')
        for n, p in sur.ego_data.items():
            traffic_vehs[n] = TrafficVeh(id=n, type='pedestrian', x=p.x, y=p.y, phi=p.phi,
                                         u=p.u)
        t1 = time.time()
        if i == 55:
            sur.ego_data['ped0'].target_points = np.array([[8.0, 4.0, 1.70, 1],
                                                           [7.9, 25.0, 1.80, 0]])
            sur.ego_data['ped0'].size = 2
        sur_list = sur.update(update=SurrUpdate(traffic_vehs=traffic_vehs))
        for value in sur.ego_data.values():
            plot_ped_data[i].append([value.x, value.y])
        for ped in sur_list.surr.ped:
            sur.ego_data[ped.id].update(x=ped.x, y=ped.y, u=ped.u, phi=ped.phi)
            sur.ego_data[ped.id].set_v()
            sur.ego_data[ped.id].set_coordinate()
            sur.ego_data[ped.id].history_x.append(ped.x)
            sur.ego_data[ped.id].history_y.append(ped.y)
            sur.ego_data[ped.id].history_u.append(ped.u)
            sur.ego_data[ped.id].history_phi.append(ped.phi)
        # if i == 50:
        #     sur.ego_data['ped0'].target_points = np.array([[8.0, 4.0, 1.70, 1],
        #                                                    [7.9, 25.0, 1.80, 0]])
            # sur.del_participant(ped)
        traffic_vehs.clear()
        t2 = time.time()
        t.append(t2 - t1)
    if if_plot:
        # plot_veh_data = plot_veh_data(veh_file_path)
        fig = plt.figure(figsize=(10, 8))
        plt.ion()
        ax = fig.add_subplot()
        for i in range(len(plot_ped_data)):
            plt.cla()
            for ped in plot_ped_data[i]:
                circle = patches.Circle(xy=(ped[0], ped[1]), radius=0.20, fc='red', ec='red', zorder=3)
                ax.add_patch(circle)
            for i in range(16):
                r = patches.Rectangle((5.0, 5.0 + i * 1.0), 5.0, 0.55, linewidth=1, edgecolor='black', facecolor='grey',
                                      alpha=0.5, zorder=2)
                ax.add_patch(r)
            # if i in list(plot_veh_data.keys()):
            #     for veh in plot_veh_data.get(i):
            #         plot_vehicle(veh[0], veh[1], veh[2])
            ax.axis('equal')
            ax.set_xlim(-5.0, 35.0)
            ax.set_ylim(-5.0, 35.0)
            plt.draw()
            plt.pause(0.1)
        plt.show()
        plt.ioff()
        plt.clf()
