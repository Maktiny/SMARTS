import os
import argparse

import gym
import matplotlib.pyplot as plt

from pathlib import Path
import datetime
import random

from smarts.core.agent import AgentPolicy
from smarts.core.agent import AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import Waypoints, NeighborhoodVehicles, AgentInterface
import network, policy
import sys, os


sys.path.append(os.pardir)
from common_space import get_submission_num, observation_adapter ,reward_adapter, action_adapter
from tool import obs_transform, observations_transform 

WORK_SPACE = os.path.dirname(os.path.realpath(__file__))

v_function_value = {} 
logp_value = {}
actions = {}

agent_index = ["AGENT-0","AGENT-1","AGENT-2","AGENT-3"]

MAX_STEP = 400

class sample_actions(AgentPolicy):
    def act(self,state):
        observations, PPO, id_list =  state["observations"],state["PPO"],state["id_list"]
        
        #obs ,image = obs_transform(obs)
        observations , images = observations_transform(observations)
        action, v_function_vale , log_P  = PPO.choose_action(observations , images)
        for id in id_list:
            v_function_value[id] = v_function_vale[id_list.index(id)]
        
            logp_value[id] = log_P[id_list.index(id)]
            #print("AAAAAAAAAAAAAAAAAAAAAAAAA")
            #print(action[id_list.index(id)][0])
            actions[id] = action[id_list.index(id)]
            actions[id][1] = action[id_list.index(id)][1] * 0.5
            #print(actions[id][0])

        return actions


def parse_args():
    parser = argparse.ArgumentParser(
        "Simple multi-agent case with lane following control."
    )
    parser.add_argument(
        "--scenario", type=str, help="Path to scenario",
    )
    parser.add_argument(
        "--horizon", type=int, default=400,
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(_args):
    scenario_path = Path(args.scenario).absolute()
    mission_num = get_submission_num(scenario_path)

    nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = "/home/carserver2/SMARTS/multi_merge/temp/data/"+ nowtime
    os.makedirs(path)
    path = os.path.join(path, "log.txt")
    


    if mission_num == -1:
        mission_num = 1

    AGENT_IDS = [f"AGENT-{i}" for i in range(mission_num)]

    agent_interface = AgentInterface(
        max_episode_steps=400,
        waypoints=Waypoints(lookahead=30),
        neighborhood_vehicles=NeighborhoodVehicles(radius=100),
        ogm=False,
        rgb=True,
        lidar=False,
        action=ActionSpaceType.Continuous,
      )

    agent_specs = [
        AgentSpec(  
            interface=agent_interface, 
            policy_builder=sample_actions,
            observation_adapter=observation_adapter,
            reward_adapter=reward_adapter,
            action_adapter=action_adapter,
            )
        for _ in range(mission_num)
    ]


    agents = dict(zip(AGENT_IDS, agent_specs))

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs=agents,
        headless=_args.headless,
        visdom=False,
        seed=42,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agents.items()}

    #import webbrowser
    #webbrowser.open('http://localhost:8081/')
    


    ########################
    #构建PPO
    ########################
    Actor_Critic = network.ActorCritic()
    PPO = policy.PPO(Actor_Critic,  agent_number=mission_num) # load the model ,set the parameter resume_model=Ture
    
    for ie in range(1000000):
        
        step = 0
        print(f"\n---- Starting episode: {ie}...")
        observations = env.reset()
        
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            step += 1
            actions = agents[random.choice(list(observations.keys()))].act({"observations":observations, "PPO":PPO, "id_list":list(observations.keys())})
            agent_actions = {
                _id: actions[_id] for _id, action in actions.items()
            }
            
            observations, rewards, dones, info = env.step(agent_actions)
            #print(observations)
            

            
            #################
            #存储experience
            ################
            print("DDDDDDDDDD")
            print(list(observations.keys()))
            print(mission_num)
            for key in list(observations.keys()):
                obs = observations[key]
                obs , image = obs_transform(obs)
                action = actions[key]
                reward = rewards[key]
                print("TTTTTTTTTTTTTTTTTTTTTT")
                print("agent:{0},agent__step:{1}, action:{2}".format(key, step, action))
                v_function = v_function_value[key]
                logp = logp_value[key]
                
                PPO.buff_store(obs, image, action, reward, v_function, logp, agent_index= agent_index.index(key))
                total_reward += reward
    
            if (step + 1) % 10 == 0:
                print(f"* Episode: {ie} * step: {step} * acc-Reward: {total_reward}")
            if MAX_STEP == step:
                break
            
        for i in list(observations.keys()):
            PPO.compute_advantage_value(v_function_value[i], agent_index.index(i))
        
        PPO.update()######ppo更新
        ###########save the model
        if ie % 50 == 0 and ie is not 0:
            PPO.save_train_model(itr=ie)

        ##save log
        
        with open(path ,"a") as f:
            f.write(f"Episode: {ie},Reward: {total_reward}\n")
            
        
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
