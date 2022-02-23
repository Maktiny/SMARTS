import math
import sys
import gym
<<<<<<< HEAD
import random
from numpy.core.fromnumeric import mean
from numpy.core.numeric import indices
# import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from sys import path 
path.append("./utils")

import utils.geometry as geometry
=======
# import cv2
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09

ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
VERSION = sys.version_info.major
if VERSION == 2:
    import cv2
elif ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)
    import cv2
    sys.path.append(ROS_PATH)
    from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import copy

from typing import Dict
from collections import namedtuple

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation.observation_function import ObservationFunction
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from smarts.core.sensors import Observation
from smarts.core.controllers import ActionSpaceType


Config = namedtuple(
    "Config", "name, agent, interface, policy, learning, other, trainer, spec"
)

<<<<<<< HEAD
global mean_speed 

=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09

SPACE_LIB = dict(
    distance_to_center=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
    heading_errors=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    speed=lambda shape: gym.spaces.Box(low=-330.0, high=330.0, shape=shape),
    steering=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
    goal_relative_pos=lambda shape: gym.spaces.Box(low=-1e2, high=1e2, shape=shape),
    neighbor=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
<<<<<<< HEAD
    #min_distance=lambda shape: gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
    img_gray=lambda shape: gym.spaces.Box(low=0.0, high=1.0, shape=shape),
)


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
<<<<<<< HEAD
    
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
<<<<<<< HEAD
        
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    
=======
        if dist < groups[i][1]:
            groups[i] = (v, dist)
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09

    return groups


class ActionSpace:
    @staticmethod
    def from_type(action_type: int):
        space_type = ActionSpaceType(action_type)
        if space_type == ActionSpaceType.Continuous:
            return gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
        elif space_type == ActionSpaceType.Lane:
<<<<<<< HEAD
            
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
            return gym.spaces.Discrete(4)
        else:
            raise NotImplementedError


class EasyOBSFn(ObservationFunction):
    @staticmethod
    def filter_obs_dict(agent_obs: dict, agent_id):
        res = copy.copy(agent_obs)
        res.pop(agent_id)
        return res

    @staticmethod
    def filter_act_dict(policies):
        return {_id: policy.action_space for _id, policy in policies}

    def __call__(self, agent_obs, worker, base_env, policies, episode, **kw):
        return {
            agent_id: {
                "own_obs": obs,
                **EasyOBSFn.filter_obs_dict(agent_obs, agent_id),
                **{f"{_id}_action": 0.0 for _id in agent_obs},
            }
            for agent_id, obs in agent_obs.items()
        }


class CalObs:
    """ Feature engineering for Observation, feature by feature.
    """

    @staticmethod
    def cal_goal_relative_pos(env_obs: Observation, **kwargs):
        ego_pos = env_obs.ego_vehicle_state.position[:2]
        goal_pos = env_obs.ego_vehicle_state.mission.goal.positions[0]

        vector = np.asarray([goal_pos[0] - ego_pos[0], goal_pos[1] - ego_pos[1]])
        space = SPACE_LIB["goal_relative_pos"](vector.shape)
        return vector / (space.high - space.low)

    @staticmethod
    def cal_distance_to_center(env_obs: Observation, **kwargs):
        """ Calculate the signed distance to the center of the current lane.
        """

        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_to_center / lane_hwidth

        dist = np.asarray([norm_dist_from_center])
        return dist

    @staticmethod
    def cal_heading_errors(env_obs: Observation, **kwargs):
        look_ahead = kwargs["look_ahead"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_path = waypoint_paths[closest_wp.lane_index][:look_ahead]

        heading_errors = [
            math.sin(math.radians(wp.relative_heading(ego.heading)))
            for wp in closest_path
        ]

        if len(heading_errors) < look_ahead:
            last_error = heading_errors[-1]
            heading_errors = heading_errors + [last_error] * (
                look_ahead - len(heading_errors)
            )
<<<<<<< HEAD
        #print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        #print(np.asarray(heading_errors).shape)
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09

        return np.asarray(heading_errors)

    @staticmethod
    def cal_speed(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res / 120.0

    @staticmethod
    def cal_steering(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])

    @staticmethod
    def cal_neighbor(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
<<<<<<< HEAD
        #print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        #print(env_obs.ego_vehicle_state.mission.start)
        #print(env_obs.ego_vehicle_state.mission.goal)
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        features = np.zeros((closest_neighbor_num, 5))
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )

        heading_angle = math.radians(ego.heading + 90.0)
        ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                continue
            v = v[0]
            rel_pos = np.asarray(
                list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
            )

            rel_dist = np.sqrt(rel_pos.dot(rel_pos))
            v_heading_angle = math.radians(v.heading)
            v_heading_vec = np.asarray(
                [math.cos(v_heading_angle), math.sin(v_heading_angle)]
            )

            ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
            rel_pos_norm_2 = rel_pos.dot(rel_pos)
            v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)
            ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
                ego_heading_norm_2 + rel_pos_norm_2
            )

            v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
                v_heading_norm_2 + rel_pos_norm_2
            )

            if ego_cosin <= 0 < v_cosin:
                rel_speed = 0
            else:
                rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

            ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)

            features[i, :] = np.asarray(
                [rel_dist, rel_speed, ttc, rel_pos[0], rel_pos[1]]
            )
<<<<<<< HEAD
        #print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        #print(features.reshape((-1,)).shape)
        #print(features.reshape((-1,)))
        #print(closest_neighbor_num)
#############feature的数据结构
        '''
        [[ 9.89802046e+00 -3.52860259e+00  1.00000000e+03  9.89604250e+00
(pid=580575)    1.97868315e-01]
(pid=580575)  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)    0.00000000e+00]
(pid=580575)  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)    0.00000000e+00]
(pid=580575)  [ 7.06779781e+00  8.38320626e-02  8.43090053e+01 -7.06150704e+00
(pid=580575)    2.98134531e-01]
(pid=580575)  [ 4.43997682e+00  2.08903837e+00  2.12536873e+00 -3.36044208e+00
(pid=580575)   -2.90186547e+00]
(pid=580575)  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)    0.00000000e+00]
(pid=580575)  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)    0.00000000e+00]
(pid=580575)  [ 1.41585373e+01  0.00000000e+00  1.00000000e+03  1.38579708e+01
(pid=580575)   -2.90186547e+00]]
(pid=580575) [ 9.89802046e+00 -3.52860259e+00  1.00000000e+03  9.89604250e+00
(pid=580575)   1.97868315e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)   0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)   0.00000000e+00  0.00000000e+00  0.00000000e+00  7.06779781e+00
(pid=580575)   8.38320626e-02  8.43090053e+01 -7.06150704e+00  2.98134531e-01
(pid=580575)   4.43997682e+00  2.08903837e+00  2.12536873e+00 -3.36044208e+00
(pid=580575)  -2.90186547e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)   0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
(pid=580575)   0.00000000e+00  0.00000000e+00  0.00000000e+00  1.41585373e+01
(pid=580575)   0.00000000e+00  1.00000000e+03  1.38579708e+01 -2.90186547e+00]

        '''
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09

        return features.reshape((-1,))

    @staticmethod
    def cal_img_gray(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]

<<<<<<< HEAD

        image_rgb = env_obs.top_down_rgb
        #print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        #print(np.asarray(image_rgb)[1])
        
        return np.asarray(image_rgb)[1]
=======
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        rgb_ndarray = env_obs.top_down_rgb
        gray_scale = (
            cv2.resize(
                rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
            )
            / 255.0
        )
        return gray_scale
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09


class EasyCallbacks(DefaultCallbacks):
    """ See example from: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
    """

    def on_episode_start(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        print("episode {} started".format(episode.episode_id))
        episode.user_data["ego_speed"] = dict()
        episode.user_data["step_heading_error"] = dict()

    def on_episode_step(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        for _id, obs in episode._agent_to_last_raw_obs.items():
            if ego_speed.get(_id, None) is None:
                ego_speed[_id] = []
            if obs.get("speed", None) is not None:
                ego_speed[_id].append(obs["speed"])

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        mean_ego_speed = {
            _id: np.mean(speed_hist) for _id, speed_hist in ego_speed.items()
        }

        distance_travelled = {
            _id: np.mean(info["score"])
            for _id, info in episode._agent_to_last_info.items()
        }

        speed_list = list(map(lambda x: round(x, 3), mean_ego_speed.values()))
        dist_list = list(map(lambda x: round(x, 3), distance_travelled.values()))
        reward_list = list(map(lambda x: round(x, 3), episode.agent_rewards.values()))

        for _id, speed in mean_ego_speed.items():
            episode.custom_metrics[f"mean_ego_speed_{_id}"] = speed
        for _id, distance in distance_travelled.items():
            episode.custom_metrics[f"distance_travelled_{_id}"] = distance

        print(
            f"episode {episode.episode_id} ended with {episode.length} steps: [mean_speed]: {speed_list} [distance_travelled]: {dist_list} [reward]: {reward_list}"
        )


class ActionAdapter:
    @staticmethod
    def from_type(action_type):
        space_type = ActionSpaceType(action_type)
        if space_type == ActionSpaceType.Continuous:
            return ActionAdapter.continuous_action_adapter
        elif space_type == ActionSpaceType.Lane:
            return ActionAdapter.discrete_action_adapter
        else:
            raise NotImplementedError

    @staticmethod
    def continuous_action_adapter(model_action):
        assert len(model_action) == 3
        return np.asarray(model_action)

    @staticmethod
    def discrete_action_adapter(model_action):
        assert model_action in [0, 1, 2, 3]
        return model_action


def _update_obs_by_item(
    ith, obs_placeholder: dict, tuned_obs: dict, space_dict: gym.spaces.Dict
):
    for key, value in tuned_obs.items():
        if obs_placeholder.get(key, None) is None:
            obs_placeholder[key] = np.zeros(space_dict[key].shape)
        obs_placeholder[key][ith] = value


def _cal_obs(env_obs: Observation, space, **kwargs):
    obs = dict()
    for name in space.spaces:
        if hasattr(CalObs, f"cal_{name}"):
            obs[name] = getattr(CalObs, f"cal_{name}")(env_obs, **kwargs)
    return obs


def subscribe_features(**kwargs):
    res = dict()

    for k, config in kwargs.items():
        if bool(config):
            res[k] = SPACE_LIB[k](config)

    return res


def get_observation_adapter(observation_space, **kwargs):
    def observation_adapter(env_obs):
        obs = dict()
        if isinstance(env_obs, list) or isinstance(env_obs, tuple):
            for i, e in enumerate(env_obs):
                temp = _cal_obs(e, observation_space, **kwargs)
                _update_obs_by_item(i, obs, temp, observation_space)
        else:
            temp = _cal_obs(env_obs, observation_space, **kwargs)
            _update_obs_by_item(0, obs, temp, observation_space)
<<<<<<< HEAD
        #print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        #print(obs)
=======
>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
        return obs

    return observation_adapter


def default_info_adapter(shaped_reward: float, raw_info: dict):
    return raw_info


<<<<<<< HEAD
'''
计算car到车道线中间的距离
'''
def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]


    closest_wp = min(closest_wps,key=lambda wp:wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center


#从env_obs.waypoint_paths中随机sample几个waypoint,然后从中选最小的speed 作为wp_speed_limit
def get_speed_limit(env_obs):
    wp_paths = env_obs.waypoint_paths
    wps_len = [len(path) for path in wp_paths]
    maxlen_lane_index = np.argmax(wps_len)
    indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])

    for i in indices:
        if len(wp_paths[maxlen_lane_index]) > i:
            global_sample_wp_path = [wp_paths[maxlen_lane_index][i]]
    wp_speed_limit = np.min(np.array([wp.speed_limit for wp in global_sample_wp_path]) / 120.) * 1.065

    return wp_speed_limit



def get_lane_dis(env_obs: Observation):
    lane_min_dis = 100000.
    min_ttc = 100000.
    min_speed = 10000.

    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
    closest_neighbor_num = 8
    surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
    heading_angle = math.radians(ego.heading + 90.0)
    ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            continue
        v = v[0]
        rel_pos = np.asarray(
                list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
            )

        rel_dist = np.sqrt(rel_pos.dot(rel_pos))
        if lane_min_dis > rel_dist:
            lane_min_dis = rel_dist
        v_heading_angle = math.radians(v.heading)
        v_heading_vec = np.asarray(
                [math.cos(v_heading_angle), math.sin(v_heading_angle)]
            )

        ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
        rel_pos_norm_2 = rel_pos.dot(rel_pos)
        v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)
        ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
                ego_heading_norm_2 + rel_pos_norm_2
            )

        v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
                v_heading_norm_2 + rel_pos_norm_2
            )

        if ego_cosin <= 0 < v_cosin:
                rel_speed = 0
        else:
                rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

        ttc = min(rel_dist / max(1e-5, rel_speed), 1e3)
        if min_ttc > ttc:
            min_ttc = ttc
        
        if min_speed > rel_speed:
            min_speed = rel_speed

            

    return lane_min_dis ,min_ttc,min_speed


def get_closest_point_index(pts_arr, pts):
    distance = [euclidean(each, pts) for each in pts_arr]
    return np.argmin(distance)

def get_path_to_goal(goal, paths, start):
    goal_pos = goal.positions
    start_pos = start.position
    path_start_pts = [each[0].pos for each in paths]

    best_path_ind = get_closest_point_index(path_start_pts, start_pos)
    path = paths[best_path_ind]

    middle_point = path[int(len(path) / 2)]
    goal_lane_id = middle_point.lane_id
    goal_lane_index = middle_point.lane_index

    path_pts = [each.pos for each in path]
    end_path_ind = get_closest_point_index(path_pts, goal_pos)

    return path



def get_box(vehicle, safety_dist):

    heading = vehicle.heading
    if heading > np.pi:
        heading = heading - 2 * np.pi

    del_dist = safety_dist / 2.0
    x_dist = -del_dist * np.sin(heading)
    y_dist = del_dist * np.cos(heading)

    return geometry.Box(
        width=vehicle.bounding_box.width,
        height=vehicle.bounding_box.length + safety_dist,
        length=vehicle.bounding_box.height,  # height of vehicle
        centerX=0.0,
        centerY=0.0,
        centerZ=0.0,
        yaw=0.0,
        roll=0.0,
        pitch=heading,  # regular yaw
        translationX=vehicle.position[0] + x_dist,
        translationY=vehicle.position[1] + y_dist,
        translationZ=vehicle.position[2],
    )


def rotate2d_vector(vectors, angle):
    ae_cos = np.cos(angle)
    ae_sin = np.sin(angle)
    rot_matrix = np.array([[ae_cos, -ae_sin], [ae_sin, ae_cos]])

    vectors_rotated = np.inner(vectors, rot_matrix)
    return vectors_rotated


def clip_angle_to_pi(angle):
    while angle < -np.pi:
        angle += np.pi * 2
    while angle > np.pi:
        angle -= np.pi * 2
    return angle


def ego_social_safety(
    agent_obs,
    d_min_ego=0.01,
    t_c_ego=0.5,
    d_min_social=0.01,
    t_c_social=0.5,
    ignore_vehicle_behind=False,
):
    # coordinate of this thing:
    # position: [x, y, z]
    # x increase when go from left to right
    # y increase when go from bottom to top
    # angle increase when vehicle move counterclockwise direction with 0 radian = bottom to top direction
    # so this is just standard coordinate shifted by pi/2 radian counterclockwise
    # add pi/2 radian to every heading to obtain the heading in standard coordinate

    def get_relative_position_vector_angle(v1, v2):
        x = v2.position[0] - v1.position[0]
        y = v2.position[1] - v1.position[1]
        # angle = clip_angle_to_pi(np.arctan2(y, x) + np.pi / 2)
        angle = clip_angle_to_pi(np.arctan2(x, y))
        return angle

    def is_behind(ego_heading_relative_diff, threshold=np.pi / 8):
        return abs(ego_heading_relative_diff) < threshold

    def get_vehicles_not_behind(ego, socials):
        ego_angle = ego.heading
        relative_position_vector_angles = [
            -get_relative_position_vector_angle(e, ego) for e in socials
        ]
        ego_heading_relative_diffs = [
            clip_angle_to_pi(ego_angle - e) for e in relative_position_vector_angles
        ]
        idxs = [
            e
            for e in range(len(socials))
            if not is_behind(ego_heading_relative_diffs[e])
        ]
        vehicles_not_behind = [socials[e] for e in idxs]
        return vehicles_not_behind

    # for debugging, you can use visualize_social_safety() in scenarios/visualization.py

    neighborhood_vehicle_states = agent_obs.neighborhood_vehicle_states
    ego_vehicle_state = agent_obs.ego_vehicle_state
    if ignore_vehicle_behind:
        neighborhood_vehicle_states = get_vehicles_not_behind(
            ego_vehicle_state, neighborhood_vehicle_states
        )

    vehicles_bounding_boxes = []
    vehicles_bounding_boxes_safety = []

    safety_dist_ego = d_min_ego + ego_vehicle_state.speed * t_c_ego

    ego_bounding_box_safety = get_box(ego_vehicle_state, safety_dist=safety_dist_ego)
    ego_bounding_box = get_box(ego_vehicle_state, safety_dist=0.0)

    for vehicle in neighborhood_vehicle_states:
        safety_dist_social = d_min_social + vehicle.speed * t_c_social
        vehicles_bounding_boxes.append(get_box(vehicle, safety_dist=0.0))
        vehicles_bounding_boxes_safety.append(
            get_box(vehicle, safety_dist=safety_dist_social)
        )

    ego_num_violations = 0
    social_num_violations = 0
    for vehicle, vehicle_safety in zip(
        vehicles_bounding_boxes, vehicles_bounding_boxes_safety
    ):
        if ego_bounding_box_safety.intersects(vehicle):
            ego_num_violations += 1
        if ego_bounding_box.intersects(vehicle_safety):
            social_num_violations += 1

    return ego_num_violations, social_num_violations


def get_relative_pos(waypoint, ego_pos):
    return [waypoint.pos[0] - ego_pos[0], waypoint.pos[1] - ego_pos[1]]

def get_closest_waypoint(ego_position, ego_heading, num_lookahead, goal_path):
    closest_wp = min(goal_path, key=lambda wp: wp.dist_to(ego_position))
    min_dist = float("inf")
    min_dist_idx = -1
    for i, wp in enumerate(goal_path):

        if wp.dist_to(ego_position) < min_dist:
            min_dist = wp.dist_to(ego_position)
            min_dist_idx = i
            closest_wp = wp

    waypoints_lookahead = [
        get_relative_pos(wp, ego_position)
        for wp in goal_path[
            min_dist_idx : min(min_dist_idx + num_lookahead, len(goal_path))
        ]
    ]
    if len(waypoints_lookahead) > 0:
        while len(waypoints_lookahead) < num_lookahead:
            waypoints_lookahead.append(waypoints_lookahead[-1])
    else:
        waypoints_lookahead = [
            get_relative_pos(closest_wp.pos, ego_position) for i in range(num_lookahead)
        ]

    waypoints_lookahead = rotate2d_vector(waypoints_lookahead, -ego_heading)
    return closest_wp, waypoints_lookahead






def get_reward_adapter(observation_adapter,**kwargs):
    def reward_adapter(observation, reward):
        env_reward  = reward  
        ego_events = observation.events
        ego_observation = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )

        

        # Distance to goal
        ego_2d_position = ego_observation.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.positions)

        closest_wp, _ = get_closest_waypoint(
            num_lookahead= 50,
            goal_path=path,
            ego_position=ego_observation.position,
            ego_heading=ego_observation.heading,
        )
        angle_error = closest_wp.relative_heading(
            ego_observation.heading
        )  # relative heading radians [-pi, pi]

        # Distance from center
        signed_dist_from_center = closest_wp.signed_lateral_error(
            observation.ego_vehicle_state.position
        )
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
        ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
        ego_speed_reward = min(
            0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
        )  # m/s
        ego_collision = len(ego_events.collisions) > 0
        ego_collision_reward = -1.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
        ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        social_safety_reward = -0.02 if social_num_violations > 0 else 0
        
        env_reward /= 10
        # DG: Different speed reward
        ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

        rewards = sum(
            [
                ego_goal_reward,
                ego_collision_reward,
                ego_off_road_reward,
                ego_off_route_reward,
                ego_wrong_way,
                ego_speed_reward,
                # ego_time_out,
                ego_dist_center_reward,
                ego_angle_error_reward,
                ego_reached_goal,
                ego_step_reward,
                env_reward,
                ego_safety_reward,
                social_safety_reward,
            ]
        )
        return rewards
    return reward_adapter
    


'''
        
#做reward shaping
def get_reward_adapter(observation_adapter, **kwargs):
    def reward_adapter(env_obs, env_reward):
        
        distance_from_center = get_distance_from_center(env_obs)
        center_penalty = -np.abs(distance_from_center)
        crash_penalty = 0.
        reach_goal_reward = 0.
        
    
        speed_penalty = 0.
        distance_penalty = 0.
        safe_distance = 0.
        #distance_ratio =0.
        lan_dis,min_ttc,min_speed = get_lane_dis(env_obs)
        

        if env_obs.events.reached_goal:
            reach_goal_reward = 500.
        
        if env_obs.events.collisions:
            crash_penalty += -500.
        
        
        if env_obs.events.off_road or env_obs.events.off_route:
            crash_penalty += -250.
        
        steering_penalty = -2 * env_obs.ego_vehicle_state.steering
        
    
        

        ###每个agent最多迭代300个step, 除以300, 避免
        if (lan_dis / env_obs.ego_vehicle_state.speed) > min_ttc:
            speed_penalty = -0.01 * env_obs.ego_vehicle_state.speed 
        #distance_ratio = env_obs.ego_vehicle_state.speed / ( get_speed_limit(env_obs=env_obs) * 0.8)
        #safe_distance = 0.03 + distance_ratio * 0.01
        safe_distance = 0.03 + (0.1 * env_obs.ego_vehicle_state.speed * 120 /15)
        distance_penalty = -0.01 * abs(safe_distance -  lan_dis) 
        

        total_reward = np.sum([2*env_reward]) + reach_goal_reward
        total_penalty = np.sum([0.01*center_penalty, 0.01*crash_penalty]) + speed_penalty + distance_penalty + steering_penalty
        #print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        #print(safe_distance)
        #print(total_reward)
        #print(speed_penalty) 
        #print(distance_penalty)

        
        return (total_penalty + total_reward) /10.
        
        return 0.1*env_reward

    return reward_adapter
'''


def get_submission_num(scenario_root):
    previous_path = sys.path.copy()
    sys.path.append(str(scenario_root))
    print(str(scenario_root))
    
=======
def get_submission_num(scenario_root):
    previous_path = sys.path.copy()
    sys.path.append(str(scenario_root))

>>>>>>> f1cbdea80b74be8e93abea99fff8f31e15544f09
    import scenario

    sys.modules.pop("scenario")
    sys.path = previous_path

    if hasattr(scenario, "agent_missions"):
        return len(scenario.agent_missions)
    else:
        return -1
