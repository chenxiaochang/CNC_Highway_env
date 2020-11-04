import numpy as np
import matplotlib
from gym.envs.registration import register

from highway_env import utils
"""A generic environment for various tasks involving a vehicle driving on a road"""
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle



class OvertakingEnv(AbstractEnv):
    """
        A two-way lane scenario modified by 'A risk management task ' with the aim to overtaking lead vehicles
        It should be satisfied the safety constraint and keep the high longitudinal velocity

        ""A risk management task: the agent is driving on a two-way lane with oncoming traffic.
        It must balance making progress by overtaking and ensuring safety.
        These conflicting objectives are implemented by a reward signal and a constraint signal,
        in the CMDP/BMDP framework.""
        """

    COLLISION_REWARD: float = -0.1  # 碰撞收益
    LEFT_LANE_CONSTRAINT: float = 1  # 左边线约束
    LEFT_LANE_REWARD: float = 0.2  # 左边线收益 the initial reward is 0.2
    HIGH_SPEED_REWARD: float = 0.8  # 高速度奖励  0.8
    RIGHT_LANE_REWARD: float = 0.3   # defined by myself

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TimeToCollision",
                "horizon": 5
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
        })
        return config

    def _reward(self, action: int) -> float:  # 该奖励设置下车辆倾向于在逆向车道行驶
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)  # 获得同一道路的所有车道索引
        # print(neighbours)  # [('a', 'b', 0), ('a', 'b', 1)]  len(neighbours)=2
        # print(self.vehicle.speed_index)  # input speed about 20,and the index=2
        reward = self.COLLISION_REWARD * self.vehicle.crashed + self.HIGH_SPEED_REWARD * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1) \
                 + self.LEFT_LANE_REWARD * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (
                         len(neighbours) - 1)+self.RIGHT_LANE_REWARD * (self.vehicle.target_lane_index[2]) / (
                         len(neighbours) - 1)
        # print(self.vehicle.SPEED_COUNT)  equal to 3
        # print(self.vehicle.target_lane_index[0])  # a
        # print(self.vehicle.target_lane_index[1])  # b
        # print(self.vehicle.target_lane_index[2])  # 0/1 ego_vehicle 所在车道

        return reward  

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed

    def _cost(self, action: int) -> float:  
        """The constraint signal is the time spent driving on the opposite lane, and occurrence of collisions."""
        return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0) / 15

    def reset(self) -> np.ndarray:
        super().reset()
        self._make_road()
        self._make_vehicles()
        return self.observation_type.observe()

    def _make_road(self, length=800):  # 建立长度800的车道
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))  # 车道1
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))  # 车道2
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.NONE)))  # 车道1处的一条反向车道

        # net.add_lane("a", "b", StraightLane([0, 8], [length, 8],
        #                                     line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE)))  # 车道1处的一条反向车道
        # net.add_lane("b", "a", StraightLane([length, 8], [0, 8],
        #                                     line_types=(LineType.NONE, LineType.NONE)))  # 车道1处的一条反向车道

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:  # 道路车辆填充
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 1)).position(30, 0),
                                                     speed=30)  # speed=30

        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # 找出车辆对应的类
        for i in range(4):  # 在此车道1上加三辆车
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))  # 得到车道索引
                              .position(70 + 40 * i + 10 * self.np_random.randn(), 0),  # 车所在的位置
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70 + 40 * i),
                              speed=24 + 2 * self.np_random.randn(),  # 每一辆车的速度
                              enable_lane_change=False)
            )
        for i in range(4):  # 在车道0上加上3辆车
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200 + 100 * i + 10 * self.np_random.randn(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200 + 100 * i),
                              speed=20 + 5 * self.np_random.randn(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)


register(
    id='overtaking-v0',
    entry_point='highway_env.envs:OvertakingEnv',
)
