import gym
import numpy as np

from ray.rllib.agents.dqn import DQNTrainer

'''
import sys
sys.path.append("/home/liyi/multi/SMARTS_Track-2-master/starter_kit")
from agent.maac.trainer import MAACTrainer
'''

from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import RGB, Waypoints, NeighborhoodVehicles

from utils import common


NAME = "DQN"
#NAME = "MAACTrainer"

def make_config(**kwargs):
    use_stacked_observation = kwargs.get("use_stacked_observation", False)
    use_rgb = kwargs.get("use_rgb", False)
    closest_neighbor_num = kwargs.get("max_observed_neighbors", 8)

    img_resolution = 24
    observe_lane_num = 3
    look_ahead = 50
    stack_size = 3 if use_stacked_observation else 1
    subscribed_features = dict(
        goal_relative_pos=(stack_size, 2),
        distance_to_center=(stack_size, 1),
        speed=(stack_size, 1),
        steering=(stack_size, 1),
        heading_errors=(stack_size, look_ahead),
        neighbor=(stack_size, closest_neighbor_num * 5),  # dist, speed, ttc
        img_gray=(stack_size, img_resolution, img_resolution) if use_rgb else False,
    )
###########  0: continue Action space Box(-1.0, 1.0, (3,), float32)   1:discrete action space Discrete(4)
   


    action_space = common.ActionSpace.from_type(1)
    #print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    #print(action_space)
    observation_space = gym.spaces.Dict(
        common.subscribe_features(**subscribed_features)
    )


# XXX: There is a bug in Ray where we can only export a trained model if
#      the policy it's attached to is named 'default_policy'.
#      See: https://github.com/ray-project/ray/issues/5339
    policy_config = {
        "default_policy":(
        None,
        observation_space,
        action_space,
        dict(model=dict(custom_model_config=dict(obs_space_dict=observation_space),)),
      )
    }





    

    interface_config = dict(
        neighborhood_vehicles=NeighborhoodVehicles(radius=100),
        waypoints=Waypoints(lookahead=50), ##########################lookhead in 50 forehead waypoints
        rgb=RGB(width=256, height=256, resolution=img_resolution / 256) if use_rgb else None,
        action=ActionSpaceType.Lane,
        road_waypoints=None,
        drivable_area_grid_map=None,
        ogm=None,
        lidar=None,
        debug=False,
    )

    observation_adapter = common.get_observation_adapter(
        observation_space,
        look_ahead=look_ahead,
        observe_lane_num=observe_lane_num,
        resize=(img_resolution, img_resolution),
        closest_neighbor_num=closest_neighbor_num,
    )

    agent_config = dict(
        observation_adapter=observation_adapter,
        reward_adapter= common.get_reward_adapter(observation_adapter),
        action_adapter=common.ActionAdapter.from_type(1),
        info_adapter=common.default_info_adapter,
    )

    learning_config = dict()

    other_config = dict(
        stop={"training_iteration": 100000}, #设置迭代次数
        checkpoint_freq=25,
        checkpoint_at_end=True,
        max_failures=1,
    )

    return common.Config(
        name=NAME,
        agent=agent_config,
        interface=interface_config,
        policy=policy_config,
        learning=learning_config,
        other=other_config,
        trainer=DQNTrainer ,
        spec={"obs": observation_space, "act": action_space},
    )
