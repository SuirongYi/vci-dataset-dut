import numpy as np
from dataclasses import dataclass
from math import pi


@dataclass
class Setting:
    # ------------仿真参数--------------------
    tau: float = 0.03  # 仿真步长
    if_sumo_test: bool = False  # 是否用sumo测试，默认为False
    # 社会力参数
    A_ped: float = 0.85
    B_ped: float = 1.95
    A_veh: float = 3.50
    B_veh: float = 0.40
    target_v: float = 1.4  # 期望速度
    max_v: float = 1.8  # 最大速度
    view_radius: float = 8.0  # 感知半径
    view_angle: float = 5 * pi / 6  # 感知角度
    adjust_time: float = 0.55  # 调整时间
    pre_horizon: int = 10  # 预测时域
    # --------------sumo仿真信息-------------------
    traffic_setting = dict(sumo_conf_path="\\sumofile\\crossing.sumocfg",
                           step_length=100)

    # --------------初始化自车信息-------------------
    target_points0 = np.array([[13.10, -11.60, 1.35],
                              [13.10, 11.60, 1.35],
                              [12.68, 74.39, 1.35]])
    target_points1 = np.array([[10.63, -15.17, 2.5],
                               [10.58, 10.45, 2.5],
                               [-15.44, 10.50, 2.5],
                               [-25.69, 10.50, 2.5]])
    target_points2 = np.array([[28.0, 10.0, 2.0],
                               [28.0, 20.0, 2.0],
                               [15.0, 20.0, 2.0],
                               [0.0, 20.0, 2.0]])
    target_points3 = np.array([[10.63, -15.17, 2.5],
                               [0.0, 0.0, 2.5],
                               [-15.44, 10.50, 2.5],
                               [-35.69, 10.50, 2.5]])
    control_object = dict(
                          ped0=dict(type='pedestrian', original_x=12.36, original_y=-28.15, u=0.0,
                                    phi=pi / 2, target_points=target_points0,
                                    adjust_time=0.95, view_radius=10.0, view_angle=2 * pi, A_sur=0.55,
                                    B_sur=1.75, A_veh=1.75, B_veh=0.55, r_x_max=8.50, pre_horizon=0.8),
                          bic0=dict(type='bicycle', original_x=10.63, original_y=-45.17, u=0.0,
                                    phi=pi / 2, target_points=target_points1,
                                    adjust_time=0.95, view_radius=15.0, view_angle=2 * pi, A_sur=0.55,
                                    B_sur=1.75, A_veh=1.60, B_veh=0.55, r_x_max=10.0, pre_horizon=0.8),
                          bic1=dict(type='bicycle', original_x=10.63, original_y=-40, u=0.0,
                                    phi=pi / 2, target_points=target_points3,
                                    adjust_time=0.95, view_radius=15.0, view_angle=2 * pi, A_sur=0.55,
                                    B_sur=1.75, A_veh=1.60, B_veh=0.55, r_x_max=10.0, pre_horizon=0.8),
                          # ped3=dict(type='pedestrian', original_x=4.0, original_y=0.0, u=0.0,
                          #           phi=pi / 2, desired_u=1.35, target_point=np.array([10.0, 20.0]),
                          #           adjust_time=0.95, view_radius=20.0, view_angle=2 * pi, A_sur=0.55,
                          #           B_sur=1.75, A_veh=1.60, B_veh=0.55, r_x_max=10.0, pre_horizon=1.0),
                          # ped4=dict(type='pedestrian', original_x=8.0, original_y=0.0, u=0.0,
                          #           phi=pi / 2, desired_u=1.35, target_point=np.array([10.0, 20.0]),
                          #           adjust_time=0.95, view_radius=20.0, view_angle=2 * pi, A_sur=0.55,
                          #           B_sur=1.75, A_veh=1.60, B_veh=0.55, r_x_max=10.0, pre_horizon=1.0),
                          )


if __name__ == "__main__":
    s = Setting()
