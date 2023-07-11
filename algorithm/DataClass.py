from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrafficVeh:
    id: str = 'default0'
    type: str = 'pedestrian'
    length: float = 0.5
    width: float = 1.0
    x: float = 0.0
    y: float = 0.0
    u: float = 0.0
    phi: float = 0.0
    acc_lon: float = 0.0
    h_x: List = field(default_factory=list)
    h_y: List = field(default_factory=list)
    h_phi: List = field(default_factory=list)

    def update(self, **kwargs):
        """update related parameters during calculation
        """
        for key, val in kwargs.items():
            assert key in vars(self), '{} is not a class attr'.format(key)
            exec("self.{0}=val".format(key), {'self': self, 'val': val})


@dataclass
class TrafficLight:
    state: str = ' '  # 红 黄 绿 状态
    value: float = 0.0


@dataclass
class SurrInfo:
    id: str = ' '
    x: float = 0.0
    y: float = 0.0
    u: float = 0.0
    phi: float = 0.0


@dataclass
class SurrData:
    ped: List[SurrInfo] = field(default_factory=list)
    bic: List[SurrInfo] = field(default_factory=list)

    def reset(self):
        self.ped.clear()
        self.bic.clear()


@dataclass
class TrafficData:
    traffic: Dict[str, TrafficVeh] = field(default_factory=dict)


@dataclass
class SurrInit:
    _setting: Dict


@dataclass
class SurrUpdate:
    traffic_vehs: Dict[str, TrafficVeh]  # include veh, ped


@dataclass
class SurrResult:
    surr: SurrData


if __name__ == "__main__":
    tr1 = TrafficVeh(id='ped0', type='pedestrian', x=1.0, y=1.2, u=3.0, phi=2.2)
    tr2 = TrafficVeh(id='ped0', type='pedestrian', x=1.0, y=1.2, u=3.0, phi=2.2)
    tr3 = TrafficVeh(id='ped0', type='pedestrian', x=3.0, y=1.2, u=3.0, phi=2.2)