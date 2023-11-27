from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass, field
from math import cos, sin, exp, fabs, pi, copysign
from numpy import ndarray
from DataClass import TrafficVeh, SurrInfo
import random


class Region:
    def __init__(self, start, end, iid):
        self.start = start
        self.end = end
        self.id = [iid]

    def __repr__(self):
        return f"[{self.start}, {self.end}] ({self.id})"


@dataclass
class TrafficParticipant(TrafficVeh):
    # 以下属性均在交通参与者参考点坐标系下表示
    r_x: float = 0.0  # 横坐标
    r_y: float = 0.0  # 纵坐标
    r_phi: float = 0.0  # 朝向角
    r_v: ndarray = np.array([0.0, 0.0])  # 速度
    r_a: ndarray = np.array([0.0, 0.0])  # 加速度
    r_self_force: ndarray = np.array([0.0, 0.0])  # 驱动力
    r_sur_force: ndarray = np.array([0.0, 0.0])  # 周围行人/非机动车对被控对象的社会力
    r_veh_force: ndarray = np.array([0.0, 0.0])  # 车辆对被控对象的社会力
    r_x_max: float = 5.5  # 最大横向偏移距离
    r_target: ndarray = np.array([0.0, 0.0])  # 当前时刻目标点在目标点坐标系下的坐标

    veh_lat_force: List = field(default_factory=list)  # 车辆横向社会力集合
    veh_lon_force: List = field(default_factory=list)  # 车辆纵向社会力集合
    max_veh_force: float = 2.5  # 最大车辆社会力数值
    max_veh_lat_force: float = 0.0  # 当前时刻最大车辆横向社会力
    max_veh_lon_force: float = 0.0  # 当前时刻最大车辆纵向社会力
    sensitive_distance: float = 1.5  # 与车辆外轮廓的横向敏感距离，用于计算车辆横向社会力
    delta: float = 0.0  # 与车辆外轮廓的纵向敏感距离，用于计算车辆纵向社会力
    radius: float = 0.20  # 被控对象圆半径
    original_x: float = 0.0  # 目标点坐标系在大地坐标系下的横坐标
    original_y: float = 0.0  # 目标点坐标系在大地坐标系下的纵坐标
    original_phi_y: float = 0.0  # 目标点坐标系在大地坐标系下的y轴角度
    original_phi_x: float = 0.0  # 目标点坐标系在大地坐标系下的x轴角度

    target_points: ndarray = np.array([[0.0, 0.0, 0.0, 0.0]])  # 被控对象的目标点集,ref_x,ref_y,ref_v, inter(0/1)
    curr_index: int = 0  # 当前时刻的目标点在目标点集中的索引
    size: int = 0  # 目标点集的大小
    v: ndarray = np.array([0.0, 0.0])  # 大地坐标系下的速度
    a: float = 0.0  # 大地坐标系下的加速度
    # a_norm: ndarray = np.array([0.0, 0.0])  # 加速度的数值大小
    a_norm: float = 0.0  # 加速度的数值大小
    u_max: float = 0.0  # 最大速度
    a_max: float = 1.50  # 最大加速度
    target_point: ndarray = np.array([0.0, 0.0])  # 当前阶段的目标点
    desired_u: float = 0.0  # 目标速度
    adjust_time: float = 0.55  # 调整时间
    view_radius: float = 0.0  # 可视域半径（/m）
    view_angle: float = 0.0  # 可视域角度（/rad）
    pre_horizon: float = 3.0  # 车辆预测时域
    vehicle_interval: float = 0.0  # 车辆横纵向边界小于1.5视为一辆车

    # 大地坐标系在目标点坐标系下的坐标值
    geodetic_x: float = 0.0
    geodetic_y: float = 0.0
    geodetic_phi: float = 0.0

    # 可视域内的交通参与者列表
    sur_vehicle: List[TrafficVeh] = field(default_factory=list)
    sur_ped_bic: List[TrafficVeh] = field(default_factory=list)
    modify_sur_vehicle: List[TrafficVeh] = field(default_factory=list)  # 目标点坐标系下车辆位置列表
    # 社会力参数
    alpha: float = 0.0
    A_sur: float = 0.0
    B_sur: float = 0.0
    A_veh: float = 0.0
    B_veh: float = 0.0
    # 历史轨迹
    history_x: List = field(default_factory=list)
    history_y: List = field(default_factory=list)
    history_u: List = field(default_factory=list)
    history_phi: List = field(default_factory=list)
    history_a: List = field(default_factory=list)
    h_veh_lat_force: List = field(default_factory=list)
    h_veh_lon_force: List = field(default_factory=list)

    def __post_init__(self):
        if self.type == 'pedestrian':
            self.u_max = 3.00
            self.a_max = 3.00
        else:
            self.u_max = 2.0
            self.a_max = 1.8
        self.x = self.original_x
        self.y = self.original_y
        self.size = self.target_points.shape[0]
        self.curr_index = 0
        self.target_point = self.target_points[self.curr_index][:2]
        self.desired_u = self.target_points[self.curr_index][2] + random.random() * 0.5
        temp = self.target_point - np.array([self.original_x, self.original_y])
        self.r_target = np.array([0.0, np.linalg.norm(temp)])  # 目标点在目标点坐标系下坐标
        self.original_phi_y = np.arctan2(temp[1], temp[0])
        self.original_phi_x = self.original_phi_y - pi / 2
        self.v = np.array([cos(self.phi), sin(self.phi)]) * self.u
        self.r_x, self.r_y, self.r_phi = \
            translate_and_rotate(self.x, self.y, self.phi, self.original_x, self.original_y, self.original_phi_x)
        self.geodetic_x, self.geodetic_y, self.geodetic_phi = \
            translate_and_rotate(0.0, 0.0, 0.0, self.original_x, self.original_y, self.original_phi_x)
        self.r_v = np.array([cos(self.r_phi), sin(self.r_phi)]) * self.u

    def set_coordinate(self):
        self.r_x, self.r_y, self.r_phi = \
            translate_and_rotate(self.x, self.y, self.phi, self.original_x, self.original_y, self.original_phi_x)
        self.r_v = np.array([cos(self.r_phi), sin(self.r_phi)]) * self.u

    def set_veh_force(self):
        self.max_veh_lon_force = max(self.veh_lon_force, key=abs)  # 纵
        self.max_veh_lat_force = max(self.veh_lat_force, key=abs)  # 横
        self.r_veh_force = np.array([self.max_veh_lat_force, self.max_veh_lon_force])

    def set_r_sum_force(self):
        self.r_a = self.r_self_force + self.r_sur_force + self.r_veh_force
        self.a_norm = np.linalg.norm(self.r_a)

    def reset_original_point(self, index):
        self.curr_index = index
        self.target_point = self.target_points[self.curr_index][:2]
        self.desired_u = self.target_points[self.curr_index][2] + random.random() * 0.5
        self.original_x = self.target_points[self.curr_index - 1][0]  # 目标点坐标系设为当前目标点的上一个目标点
        self.original_y = self.target_points[self.curr_index - 1][1]
        temp = self.target_point - np.array([self.original_x, self.original_y])
        self.r_target = np.array([0.0, np.linalg.norm(temp)])
        self.original_phi_y = np.arctan2(temp[1], temp[0])
        self.original_phi_x = self.original_phi_y - pi / 2
        self.r_x, self.r_y, self.r_phi = \
            translate_and_rotate(self.x, self.y, self.phi, self.original_x, self.original_y, self.original_phi_x)
        self.r_v = np.array([cos(self.r_phi), sin(self.r_phi)]) * self.u
        self.geodetic_x, self.geodetic_y, self.geodetic_phi = \
            translate_and_rotate(0.0, 0.0, 0.0, self.original_x, self.original_y, self.original_phi_x)

    def set_v(self):
        self.v = np.array([cos(self.phi), sin(self.phi)]) * self.u

    def reset(self):
        self.sur_vehicle.clear()
        self.modify_sur_vehicle.clear()
        self.sur_ped_bic.clear()
        self.r_self_force = np.array([0.0, 0.0])
        self.r_sur_force = np.array([0.0, 0.0])
        self.r_veh_force = np.array([0.0, 0.0])
        self.veh_lat_force.clear()
        self.veh_lon_force.clear()
        self.max_veh_lat_force = 0.0
        self.max_veh_lon_force = 0.0


@dataclass
class Model:
    control_object: Dict[str, TrafficParticipant] = field(default_factory=dict)
    tau: float = 0.1

    def update(self) -> List[SurrInfo]:
        objet_list = list()
        for name, obj in self.control_object.items():
            calculate_veh_force(obj)
            calculate_self_force(obj)
            calculate_sur_force(obj, self.tau)
            obj.set_r_sum_force()
            if obj.a_norm > obj.a_max:
                obj.r_a = obj.r_a * (obj.a_max / obj.a_norm)
            next_v = obj.r_v + obj.r_a * self.tau
            v_norm = np.linalg.norm(next_v)
            if v_norm > obj.u_max:
                next_v = next_v * (obj.u_max / v_norm)
            next_phi = np.arctan2(next_v[1], next_v[0])
            next_position = np.array([obj.r_x, obj.r_y]) + 0.5 * (obj.r_v + next_v) * self.tau
            next_x = next_position[0]
            next_y = next_position[1]
            if fabs(next_x) <= obj.r_x_max:
                r_next_x = next_x
            else:
                r_next_x = obj.r_x
            if next_y >= obj.r_y:
                r_next_y = next_y
            else:
                r_next_y = obj.r_y
            obj_x, obj_y, obj_phi = translate_and_rotate(r_next_x, r_next_y, next_phi, obj.geodetic_x, obj.geodetic_y,
                                                         obj.geodetic_phi)
            obj.history_a.append(obj.a_norm)
            objet_list.append(SurrInfo(id=name, x=obj_x, y=obj_y, u=np.linalg.norm(next_v), phi=obj_phi))
        return objet_list

    def reset(self):
        self.control_object.clear()


def calculate_veh_force(obj: TrafficParticipant):
    veh_in_new_coordinate(obj)  # 将车辆位置转换为参考点坐标系下位置
    number = len(obj.sur_vehicle)
    if number == 0:
        pass
    elif number == 1:
        update_veh_force(obj)
    else:
        vehicle_merging(obj)  # 合并位置相近的车辆
        update_veh_force(obj)


def calculate_sur_force(obj: TrafficParticipant, tau):
    number = len(obj.sur_ped_bic)
    if number == 0:
        pass
    else:
        attention_list = []
        mixed_state = np.empty(shape=(number, 4))
        for _, o in enumerate(obj.sur_ped_bic):
            # 周围行人和非机动车坐标改为目标点坐标系下坐标
            t_x, t_y, t_phi = translate_and_rotate(o.x, o.y, o.phi, obj.original_x, obj.original_y, obj.original_phi_x)
            mixed_state[_] = np.array([t_x, t_y, o.u, t_phi])
            attention_value = obj.alpha + (1 - obj.alpha) * (1 + calculate_cos_phi(obj, o)) / 2
            attention_list.append(attention_value)
        ego_position = np.array([obj.r_x, obj.r_y])  # ego也是目标点坐标系下坐标
        curr_sur_position = mixed_state[:, :2]
        curr_sur_v = get_speed(mixed_state)
        delta_position = curr_sur_v * tau * 2.0
        pre_sur_position = curr_sur_position + delta_position
        a = curr_sur_position - ego_position
        b = pre_sur_position - ego_position
        temp1 = np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)
        temp2 = np.linalg.norm(delta_position, axis=1)
        b_beta = 0.5 * np.sqrt(np.power(temp1, 2) - np.power(temp2, 2))
        temp3 = a / np.linalg.norm(a, axis=1).reshape(number, 1)
        temp4 = b / np.linalg.norm(b, axis=1).reshape(number, 1)
        direction = -1 * (temp3 + temp4) / np.linalg.norm(temp3 + temp4, axis=1).reshape(number, 1)
        for i in range(len(direction)):
            vector1 = a[i]
            vector2 = b[i]
            theta = np.arccos(np.clip(vector1.dot(vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)), -1.0, 1.0))
            if theta <= 0.1:
                direction[i] = np.array([-direction[i][1], direction[i][0]])
        force = (obj.A_sur * np.exp(-obj.B_sur * (b_beta - 0.2))).reshape(number, 1) * direction * np.array(
            attention_list).reshape(number, 1)
        obj.update(r_sur_force=np.sum(force, axis=0))


def calculate_self_force(obj: TrafficParticipant):
    number = len(obj.sur_vehicle)
    vector = obj.r_target - np.array([obj.r_x, obj.r_y])
    unit_vector = vector / np.linalg.norm(vector)
    self_force = (obj.desired_u * unit_vector - obj.r_v) / obj.adjust_time
    lat_self_force = np.array([self_force[0], 0.0])
    lon_self_force = np.array([0.0, self_force[1]])
    if number == 0:
        obj.update(r_self_force=lon_self_force + lat_self_force)
    else:
        if fabs(obj.max_veh_lat_force) > 0.00:
            lat_self_force = np.array([0.0, 0.0])
            if copysign(1, obj.max_veh_lon_force) == -1 and fabs(obj.max_veh_lon_force) >= 1.0:  # todo
                obj.update(r_self_force=np.array([0, -2.5]) + lat_self_force)
            else:
                obj.update(r_self_force=lon_self_force + lat_self_force)
        else:
            obj.update(r_self_force=lon_self_force + lat_self_force)


def update_veh_force(obj: TrafficParticipant):
    for veh_ in obj.modify_sur_vehicle:
        # attention_value = obj.alpha + (1 - obj.alpha) * (1 + calculate_cos_phi(obj, veh_, 'reference_point')) / 2
        trans_x = obj.r_x - veh_.x
        trans_y = obj.r_y - veh_.y
        if trans_y >= 0:
            lon_factor = 1.0
        else:
            lon_factor = -1.0
        if trans_x >= 0:
            lat_factor = 1.0
        else:
            lat_factor = -1.0
        low = -max(1.5, veh_.width / 2) - veh_.width / 2       # todo 与车辆外轮廓距离，是固定值还是与车辆宽有关？
        # low = -veh_.width       # todo 与车辆外轮廓距离，是固定值还是与车辆宽有关？
        if fabs(trans_x) <= veh_.length / 2:
            value_factor = 1.0
        elif veh_.length / 2 < fabs(trans_x) <= veh_.length / 2 + obj.sensitive_distance:
            value_factor = (veh_.length / 2 + obj.sensitive_distance - fabs(trans_x)) / obj.sensitive_distance
        else:
            value_factor = 0.0
        if low <= trans_y <= veh_.width / 2:
            if -veh_.width / 2 <= trans_y <= veh_.width / 2:
                delta_d_x = fabs(trans_x) - veh_.length / 2
            else:
                delta_d_x = fabs(trans_x) - (trans_y - low) * veh_.length / (-veh_.width - 2 * low)
            lat_magnitude_ = min(obj.A_veh * exp(-obj.B_veh * (delta_d_x - obj.radius)), obj.max_veh_force) * lat_factor * value_factor
        else:
            lat_magnitude_ = 0.0
        # delta = 0.0
        temp = obj.delta + veh_.length / 2
        if -temp <= trans_x <= temp:
            if -veh_.length / 2 <= trans_x <= veh_.length / 2:
                delta_d_y = fabs(trans_y) - veh_.width / 2
            else:
                delta_d_y = fabs(trans_y) - veh_.width * ((veh_.length + 2 * obj.delta) - fabs(trans_x)) / (
                        4 * obj.delta)
            # lon_magnitude_ = obj.A_veh * exp(-obj.B_veh * delta_d_y) * lon_factor
            lon_magnitude_ = min(obj.A_veh * exp(-obj.B_veh * (delta_d_y - obj.radius)), obj.max_veh_force) * lon_factor
        else:
            lon_magnitude_ = 0.0
        obj.veh_lat_force.append(lat_magnitude_)
        obj.veh_lon_force.append(lon_magnitude_)
    obj.set_veh_force()


def veh_in_new_coordinate(obj: TrafficParticipant):
    """将车辆坐标转化为目标点坐标系下坐标"""
    for veh in obj.sur_vehicle:
        trans_x, trans_y, trans_phi = translate_and_rotate(veh.x, veh.y, veh.phi, obj.original_x,
                                                           obj.original_y, obj.original_phi_x)
        pre_trans_x = trans_x + veh.u * cos(trans_phi) * obj.pre_horizon
        pre_trans_y = trans_y + veh.u * sin(trans_phi) * obj.pre_horizon
        x_min = np.min(np.array([get_veh_boundary(TrafficVeh(x=trans_x, y=trans_y, phi=trans_phi, length=veh.length,
                                                             width=veh.width))[0] +
                                 get_veh_boundary(TrafficVeh(x=pre_trans_x, y=pre_trans_y, phi=trans_phi,
                                                             length=veh.length, width=veh.width))[0]]))
        x_max = np.max(np.array([get_veh_boundary(TrafficVeh(x=trans_x, y=trans_y, phi=trans_phi, length=veh.length,
                                                             width=veh.width))[0] +
                                 get_veh_boundary(TrafficVeh(x=pre_trans_x, y=pre_trans_y, phi=trans_phi,
                                                             length=veh.length, width=veh.width))[0]]))
        y_min = np.min(np.array([get_veh_boundary(TrafficVeh(x=trans_x, y=trans_y, phi=trans_phi, length=veh.length,
                                                             width=veh.width))[1] +
                                 get_veh_boundary(TrafficVeh(x=pre_trans_x, y=pre_trans_y, phi=trans_phi,
                                                             length=veh.length, width=veh.width))[1]]))
        y_max = np.max(np.array([get_veh_boundary(TrafficVeh(x=trans_x, y=trans_y, phi=trans_phi, length=veh.length,
                                                             width=veh.width))[1] +
                                 get_veh_boundary(TrafficVeh(x=pre_trans_x, y=pre_trans_y, phi=trans_phi,
                                                             length=veh.length, width=veh.width))[1]]))
        obj.modify_sur_vehicle.append(TrafficVeh(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2, phi=0.0,
                                                 width=y_max - y_min, length=x_max - x_min))
        # veh.update(x=(x_min + x_max) / 2, y=(y_min + y_max) / 2, phi=0.0, width=y_max - y_min, length=x_max - x_min)


def vehicle_merging(obj: TrafficParticipant):
    """将有外轮廓相近或者重叠的车辆合并"""
    regions_x = []
    new_veh_list = []
    for i, v in enumerate(obj.modify_sur_vehicle):
        regions_x.append(Region(start=v.x - v.length / 2, end=v.x + v.length / 2, iid=i))
    result_x = merge_regions(regions_x, obj.vehicle_interval)  # 返回在x轴上有重叠或很近的车辆编号
    for region in result_x:
        regions_y = []
        for vid in region.id:
            veh = obj.modify_sur_vehicle[vid]
            regions_y.append(Region(start=veh.y - veh.width / 2, end=veh.y + veh.width / 2, iid=vid))
        result_y = merge_regions(regions_y, obj.vehicle_interval)
        for region_ in result_y:
            if len(region_.id) == 1:
                new_veh_list.append(obj.modify_sur_vehicle[region_.id[0]])
            else:  # 横纵向轮廓间距均小于给定数值，合并
                temp_x = []
                temp_y = []
                for vid_ in region_.id:
                    temp_x.extend(get_veh_boundary(obj.modify_sur_vehicle[vid_])[0])
                    temp_y.extend(get_veh_boundary(obj.modify_sur_vehicle[vid_])[1])
                max_x = np.max(np.array(temp_x))
                min_x = np.min(np.array(temp_x))
                max_y = np.max(np.array(temp_y))
                min_y = np.min(np.array(temp_y))
                new_veh_list.append(
                    TrafficVeh(x=(max_x + min_x) / 2, y=(max_y + min_y) / 2, phi=0.0, length=max_x - min_x,
                               width=max_y - min_y))
    obj.modify_sur_vehicle = new_veh_list


def merge_regions(regions: List[Region], interval: float) -> List[Region]:
    regions.sort(key=lambda x: x.start)  # 列表元素排序
    merged = []
    curr_regions = regions[0]
    for region in regions[1:]:
        if region.start - curr_regions.end <= interval:
            curr_regions.end = max(curr_regions.end, region.end)
            curr_regions.id.extend(region.id)  # 添加被合并的原始区间的 id
        else:
            merged.append(curr_regions)
            curr_regions = region
    merged.append(curr_regions)
    return merged


def translate_and_rotate(o_x, o_y, o_phi, v_x, v_y, v_phi):
    trans_x = (o_x - v_x) * cos(v_phi) + (o_y - v_y) * sin(v_phi)
    trans_y = -(o_x - v_x) * sin(v_phi) + (o_y - v_y) * cos(v_phi)
    trans_phi = o_phi - v_phi
    return trans_x, trans_y, trans_phi


def calculate_cos_phi(obj: TrafficParticipant, vehicle: TrafficVeh, coo_system: str = 'geodetic') -> float:
    if coo_system == 'geodetic':
        a = np.array([vehicle.x - obj.x, vehicle.y - obj.y])
        if obj.u == 0.0:
            return 0.0
        else:
            return a.dot(obj.v) / (np.linalg.norm(a) * obj.u)
    elif coo_system == 'reference_point':
        a = np.array([vehicle.x - obj.r_x, vehicle.y - obj.r_y])
        if obj.u == 0.0:
            return 0.0
        else:
            return a.dot(obj.r_v) / (np.linalg.norm(a) * obj.u)


def get_veh_boundary(vehicle: TrafficVeh) -> Tuple:
    x_a = vehicle.x + vehicle.width / 2 * sin(vehicle.phi) - vehicle.length / 2 * cos(vehicle.phi)
    y_a = vehicle.y - vehicle.width / 2 * cos(vehicle.phi) - vehicle.length / 2 * sin(vehicle.phi)
    # right_rear = np.array([x_a, y_a])
    x_b = x_a + vehicle.length * cos(vehicle.phi)
    y_b = y_a + vehicle.length * sin(vehicle.phi)
    # right_front = np.array(x_b, y_b)
    x_c = x_a - vehicle.width * np.sin(vehicle.phi)
    y_c = y_a + vehicle.width * np.cos(vehicle.phi)
    # left_rear = np.array([x_c, y_c])
    x_d = x_a + vehicle.length * np.cos(vehicle.phi) - vehicle.width * np.sin(vehicle.phi)
    y_d = y_a + vehicle.length * np.sin(vehicle.phi) + vehicle.width * np.cos(vehicle.phi)
    # left_front = np.array([x_d, y_d])
    return [x_a, x_b, x_c, x_d], [y_a, y_b, y_c, y_d]


def get_speed(state: ndarray) -> ndarray:
    number = state.shape[0]
    v = state[:, 2].reshape(number, 1)
    phi = state[:, 3].reshape(number, 1)
    con_phi = np.concatenate((np.cos(phi), np.sin(phi)), axis=1)
    return v * con_phi


def if_within_bounds(obj: TrafficParticipant, tau: float, p: int) -> bool:
    pre_obj_traj_x = []
    for j in range(p):
        pre_obj_traj_x.append(
            obj.r_x + obj.u * cos(obj.r_phi) * tau * (p + 1))
    if np.max(np.array(pre_obj_traj_x)) >= obj.r_x_max:
        return False
    else:
        return True


if __name__ == "__main__":
    ...
