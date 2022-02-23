import sys
import argparse
import ray

from pathlib import Path
from ray import tune
from ray.rllib import policy
from ray.rllib.agents.trainer import COMMON_CONFIG

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface

from utils.common import EasyCallbacks, get_submission_num
from rllib_agent_config import make_config 

RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{algorithm}-{n_agent}"


def parse_args():
    parser = argparse.ArgumentParser("Multi-agent learning")

    parser.add_argument(
        "--scenario", type=str, help="Scenario name",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument("--log_dir", type=str, default="~/ray_results/smarts/")

    return parser.parse_args()


def main(args):
    '''
    ray.init()

    print(
        "--------------- Ray startup ------------\n{}".format(
            ray.state.cluster_resources()
        )
    )
    '''

    scenario_path = Path(args.scenario).absolute()
    n_mission = get_submission_num(scenario_path)
    

    if n_mission == -1:
        raise ValueError("No mission can be found")

    config = make_config()

    agents = {
        f"AGENT-{i}": AgentSpec(
            **config.agent, interface=AgentInterface(**config.interface)
        )
        for i in range(n_mission)
    }

    env_config = {
        "seed": 42,
        "scenarios": [str(scenario_path)],
        "headless": args.headless,
        "agent_specs": agents,
        "visdom": False, #可视化
    }


    '''
    #policies = {k: config.policy for k in agents},
    tune_config = {
        "env": RLlibHiWayEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "callbacks": EasyCallbacks,
        "log_level": "WARN",
        "num_workers": 1,
        "horizon": 300,
        **config.learning,
    }
'''
######
    tune_config = {
        "env": RLlibHiWayEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": config.policy,
            
        },
        "callbacks": EasyCallbacks,
        "log_level": "WARN",
        "num_workers": 1,
        "horizon": 500,
        **config.learning,
    }

    experiment_name = EXPERIMENT_NAME.format(
        scenario=scenario_path.stem, algorithm="DQN", n_agent=len(agents)
    )

    log_dir = Path(args.log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)

###########打开浏览器,可视化
    #import webbrowser
    #webbrowser.open('http://localhost:8081/')
   

    # run experiments
    analysis = tune.run(
        config.trainer,
        **config.other,
        name=experiment_name,
        local_dir=str(log_dir),
        export_formats=["model", "checkpoint"],
        config=tune_config,
        #resume=True, #可以从上一个checkpoint接着开始训练
    )
   
    print(analysis.dataframe().head())


if __name__ == "__main__":
    args = parse_args()
    main(args)
