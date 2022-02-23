'''
Author: your name
Date: 2021-06-23 17:00:44
LastEditTime: 2021-06-28 14:34:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SMARTS/multi_merge/starter_kit/agg_merge/ppo/tool.py
'''
# -*- encoding=utf-8 -*-
import numpy as np
import os.path as osp
import os
import shutil
import torch

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
obs_len = 139
###### obs 除了image之外总共97个参数

def obs_transform(env_obs):
    
    index = ["distance_from_center","heading_errors","wp_errors","wp_speed_penalty","speed","steering","lane_ttc",
    "lane_dist","closest_lane_nv_rel_speed","intersection_ttc","intersection_distance","closest_its_nv_rel_speed",
    "closest_its_nv_rel_pos","min_dist","detect_car","waypoint","threaten_distance"]
    
    #index = ["distance_to_center" ,"heading_errors","speed","steering","start_pos", "goal_relative_pos", "neighbor", "waypoint" ]
    index = ["distance_to_center" ,"heading_errors","speed","steering","start_pos", "goal_relative_pos", "neighbor", "waypoint" ]
    obs = np.zeros(1,dtype=np.float32)
    
    for i in range(8):
        obs = np.append(obs,env_obs[index[i]],axis=0)
    obs = obs[1:].reshape(1,-1)
    image = np.array(env_obs["image_rgb"])[::4,::4]
    #handle the error 
   
    global obs_len
    if len(obs[0]) is not obs_len:
        for i in range(abs(len(obs[0]) - obs_len)):
            obs = np.append(obs,0)
    obs_len = len(obs[0])
    
    return obs, image.reshape((3,64,64))


def observations_transform(env_obs):
   
    index = ["distance_from_center","heading_errors","wp_errors","wp_speed_penalty","speed","steering","lane_ttc",
    "lane_dist","closest_lane_nv_rel_speed","intersection_ttc","intersection_distance","closest_its_nv_rel_speed",
    "closest_its_nv_rel_pos","min_dist","detect_car","waypoint","threaten_distance"]
  
    #index = ["distance_to_center" ,"heading_errors","speed","steering","start_pos", "goal_relative_pos", "neighbor","waypoint"]
 
    image = np.zeros(( 64 , 64, 3 ),dtype=np.float32)
    obs = np.zeros(1,dtype=np.float32)

    keys = list(env_obs.keys())
    
    '''
    for t in agent_index:
        if t in keys:
            for i in range(8):
                obs = np.append(obs,env_obs[t][index[i]],axis=0)
            image = np.append(image, env_obs[t]["image_rgb"][::4,::4],axis=2)
        else:
            obs = np.append(obs, obs_padding, axis=0)
            image = np.append(image, image_padding,axis=2)
    '''
    for t in keys:
        for i in range(8):
            obs = np.append(obs,env_obs[t][index[i]],axis=0)
        image = np.append(image, env_obs[t]["image_rgb"][::4,::4],axis=2)
    #print(image.shape)
    image = image.reshape(len(keys) + 1 , 3 , 64, 64)
    
    return obs[1:].reshape(1,-1), image


def save_torch_model(path, model, itr=None):
    file_path = path +"train_model"+("%d"%itr if itr is not None else "")
    if osp.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)
    save_file = osp.join(file_path, "model.pkl")
    torch.save(model.state_dict(), save_file)
    print("model saved in path:%s"% save_file)

def reload_torch_model(path, model, itr=None):
    #file_path = path +"train_model"+("%d"%itr if itr is not None else "")
    file_path = "../temp/data/2021-06-29-11-04-54train_model350"
    save_file = osp.join(file_path, "model.pkl")
    model.load_state_dict(torch.load(save_file))
    model.eval()
    print("model reload from path:%s"% save_file)




    




