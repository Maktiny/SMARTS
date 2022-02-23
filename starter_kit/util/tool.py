# -*- encoding=utf-8 -*-
import numpy as np

'''
 return {
        "distance_to_center" : np.array([distance_to_center]),
        "heading_errors" : np.array([heading_errors]),
        "speed" : np.array([speed]),
        "steering" : np.array([steering]),
        "start_pos":np.array([start_pos]),
        "goal_relative_pos" : np.array([goal_relative_pos]),
        "neighbor" : np.array([neighbor]),
        "image_rgb" : np.array([image_rgb])
    }

'''
###### obs 除了image之外总共97个参数

def obs_transform(env_obs):
    index = ["distance_to_center","heading_errors","speed","steering","start_pos","goal_relative_pos","neighbor"]
    obs = np.zeros(97,dtype=np.float32)
    image = np.zeros(256*256*3,dtype=np.float32)
    for i in range(len(env_obs) - 1 ):
        obs = np.append(obs,env_obs[index[i]],axis=0)
    image = np.append(image, env_obs["image_rgb"], axis=0)
    return obs, image


def observations_transform(env_obs):
    index = ["distance_to_center","heading_errors","speed","steering","start_pos","goal_relative_pos","neighbor"]
    image = np.zeros(256 * 256 * 3 * len(env_obs),dtype=np.float32)
    obs = np.zeros(97,dtype=np.float32)
    for t in range(len(env_obs)):
        for i in range(7):
            obs = np.append(obs,env_obs[index[i]],axis=0)
        image = np.append(image, env_obs["image_rgb"], axis=0)
    return obs, image



