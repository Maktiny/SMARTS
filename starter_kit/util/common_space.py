
import gym
import numpy as np
import math
import sys

from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from smarts.core.sensors import Observation
from sys import path 
path.append("./util")

import util.geometry as geometry

#########################################
# parameters seting
#########################################

config = {
    "look_ahead" : 50,
    "closest_neighbor_num" : 8
}




# ==================================================
# Continous Action Space
# throttle, brake, steering
# ==================================================

ACTION_SPACE = gym.spaces.Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
)

def action_adapter():
    return ACTION_SPACE


# ==================================================
# Observation Space
# This observation space should match the output of observation(..) below
# ==================================================
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        # To make car follow the waypoints
        # distance from lane center
        "distance_to_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # relative heading angle from 10 waypoints in 50 forehead waypoints
        "heading_errors": gym.spaces.Box(low=-1.0, high=1.0, shape=(50,)),

        # Car attributes
        # ego speed
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # ego steering
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # To make car learn to slow down, overtake or dodge
        "start_pos":gym.spaces.Box(low=-1e2, high=1e2, shape=(2,)),
        "goal_relative_pos": gym.spaces.Box(low=-1e2, high=1e2, shape=(2,)),
        # the ibformation of ego_vechiel's neighbor car, just 5
        "neighbor": gym.spaces.Box(low=-1e3, high=1e3, shape=(40,)),
        "img_rgb": gym.spaces.Box(low=0.0, high=255.0, shape=(2,)), # 256 x 256 x 3 一张图片
       
    }
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
    
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    

    return groups


def cal_start_pos(env_obs:Observation):
    return env_obs.ego_vehicle_state.mission.start.position


def cal_goal_relative_pos(env_obs: Observation):
        ego_pos = env_obs.ego_vehicle_state.position[:2]
        goal_pos = env_obs.ego_vehicle_state.mission.goal.positions[0]

        vector = np.asarray([goal_pos[0] - ego_pos[0], goal_pos[1] - ego_pos[1]])
        return vector.shape



def cal_distance_to_center(env_obs: Observation):
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


def cal_heading_errors(env_obs: Observation):
        look_ahead = config["look_ahead"]
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

        return np.asarray(heading_errors)


def cal_speed(env_obs: Observation):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res / 120.0



def cal_steering(env_obs: Observation):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])



def cal_neighbor(env_obs: Observation):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        #print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        #print(env_obs.ego_vehicle_state.mission.start)
        #print(env_obs.ego_vehicle_state.mission.goal)
        closest_neighbor_num = config["closest_neighbor_num"]
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
        #print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        #print(features)
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

        return features.reshape((-1,))
    

def cal_img_rgb(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]

        image_rgb = env_obs.top_down_rgb
        
        
        return np.asarray(image_rgb)[1]


####################################
####   observation_adapter
###################################
def observation_adapter(env_obs):
    distance_to_center = cal_distance_to_center(env_obs)
    heading_errors = cal_heading_errors(env_obs)
    speed = cal_speed(env_obs)
    steering = cal_steering(env_obs)
    start_pos = cal_start_pos(env_obs)
    goal_relative_pos = cal_goal_relative_pos(env_obs)
    neighbor = cal_neighbor(env_obs)
    image_rgb = cal_img_rgb(env_obs)

    return {
        "distance_to_center" : np.array(distance_to_center),
        "heading_errors" : np.array(heading_errors),
        "speed" : np.array(speed),
        "steering" : np.array(steering),
        "start_pos":np.array(start_pos),
        "goal_relative_pos" : np.array(goal_relative_pos),
        "neighbor" : np.array(neighbor),
        "image_rgb" : np.array(image_rgb)
    }



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

####################################
####   reward_adapter
###################################

def reward_adapter(observation, reward):
    env_reward = reward
    ego_events = observation.events
    ego_observation = observation.ego_vehicle_state
    start = observation.ego_vehicle_state.mission.start
    goal = observation.ego_vehicle_state.mission.goal
    path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )

    #linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
    #angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

        # Distance to goal
    ego_2d_position = ego_observation.position[0:2]
    goal_dist = distance.euclidean(ego_2d_position, goal.positions)

    closest_wp, _ = get_closest_waypoint(
            num_lookahead= config["look_ahead"],
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
    ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
    #ego_linear_jerk = -0.0001 * linear_jerk
    #ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
    env_reward /= 100
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
                #ego_linear_jerk,
                #ego_angular_jerk,
                ego_lat_speed,
                ego_safety_reward,
                social_safety_reward,
            ]
        )
    return rewards

       




def get_submission_num(scenario_root):
    previous_path = sys.path.copy()
    sys.path.append(str(scenario_root))
    print(str(scenario_root))
    
    import scenario

    sys.modules.pop("scenario")
    sys.path = previous_path

    if hasattr(scenario, "agent_missions"):
        return len(scenario.agent_missions)
    else:
        return -1