import argparse
from gym.spaces import Tuple, MultiDiscrete

import ray
from ray import tune
from ray.tune import register_env
from agents.maac import MAACTrainer
from smac.examples.rllib.env import RLlibStarCraft2Env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=5000000)
    parser.add_argument("--map-name", type=str, default="3m")
    args = parser.parse_args()

    def env_creator(smac_args):
        env = RLlibStarCraft2Env(**smac_args)
        agent_list = list(range(env._env.n_agents))
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for i in agent_list])
        # act_space = Tuple([env.action_space for i in agent_list])
        act_space = MultiDiscrete([env.action_space.n for i in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space)

    ray.init(num_gpus=1)
    register_env("sc2_grouped", env_creator)

    stop = {"training_iteration": args.num_iters}

    config = {
        'env': 'sc2_grouped',
        'framework': 'torch',
        'num_gpus': 1,
        "env_config": {
            "map_name": args.map_name,
        },

    }

    tune.run(
        MAACTrainer,
        name='my_maac_sc',
        stop=stop,
        config=config,
        metric='episode_reward_mean',
    )
