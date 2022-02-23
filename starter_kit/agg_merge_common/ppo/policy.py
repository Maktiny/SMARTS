import re
import numpy as np
from numpy.core.numeric import argwhere
from numpy.lib.function_base import select
import scipy.signal
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
import sys, os


sys.path.append(os.pardir)
from tool import  save_torch_model,reload_torch_model


'''   state_dim:
{'distance_to_center': array([[-0.00835291]]), 'goal_relative_pos': array([[ 0.69721077, -0.00731   ]]), 'heading_errors': array([[ 3.33498377e-05,  3.33498377e-05, -1.37811385e-04,
(pid=49023)         -1.37811385e-04, -1.37811385e-04, -1.37811385e-04,
(pid=49023)         -1.37811385e-04,  6.07530305e-04,  6.07530305e-04,
(pid=49023)          6.07530305e-04,  6.07530305e-04,  1.65656504e-03,
(pid=49023)          1.65656504e-03,  1.65656504e-03,  1.65656504e-03,
(pid=49023)          2.82171821e-03,  2.82171821e-03,  2.82171821e-03,
(pid=49023)          2.82171821e-03,  3.80962041e-03,  3.80962041e-03,
(pid=49023)          3.80962041e-03,  3.80962041e-03,  5.96014474e-03,
(pid=49023)          5.96014474e-03,  5.96014474e-03,  5.96014474e-03,
(pid=49023)          8.65329734e-03,  8.65329734e-03,  8.65329734e-03,
(pid=49023)          8.65329734e-03,  1.18535887e-02,  1.18535887e-02,
(pid=49023)          1.18535887e-02,  1.18535887e-02,  1.18535887e-02,
(pid=49023)          1.18535887e-02,  1.18535887e-02,  1.14112721e-02,
(pid=49023)          1.14112721e-02,  1.14112721e-02,  1.15560051e-02,
(pid=49023)          1.15560051e-02,  1.43529890e-02,  1.43529890e-02,
(pid=49023)          1.71978834e-02,  1.71978834e-02,  1.71978834e-02,
(pid=49023)          1.71978834e-02,  1.79947512e-02]]), 'neighbor': array([[   0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,    0.        ,    0.        ,
(pid=49023)            0.        ,    0.        ,   18.57853715,    0.        ,
(pid=49023)         1000.        ,   10.60822829,  -15.25213215,   97.20626889,
(pid=49023)            0.        , 1000.        ,   93.47639331,  -26.66875709]]), 'speed': array([[0.0475613]]), 'steering': array([[-0.00204455]])}
'''

config = {
    "state_dim" :139, #79,##state_dim contain 97 argumnets except img_data
    "action_dim" : 2,#throttle ,brake ,steering, [brake,throttle] = [-1,1]
    "img_batch" : 1, # just an img 
    "img_size" : [3 , 64 , 64 ],#RGN
}
import datetime
nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

path  = "../temp/data/" + nowtime

def discount_sum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class PPOBuffer:
    def __init__(self, obs_dim, img_dim, action_dim, steps, gamma=0.99, lamda = 0.95, agent_num=1):
        super(PPOBuffer,self).__init__()
        self.obs_buf = np.zeros([agent_num, steps, obs_dim], dtype=np.float32)
        self.img_buf = np.zeros([agent_num, steps, img_dim[0], img_dim[1], img_dim[2]], dtype=np.float32)
        self.action_buf = np.zeros([agent_num, steps, action_dim], dtype=np.float32)
        self.advantage_buf = np.zeros([agent_num, steps], dtype=np.float32)
        self.reward_buf = np.zeros([agent_num, steps], dtype=np.float32)
        self.discount_reward_buf = np.zeros([agent_num,steps],dtype=np.float32)
        self.v_function_buf = np.zeros([agent_num, steps], dtype=np.float32)
        self.logp_buf = np.zeros([agent_num, steps], dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda 

        self.agent_step, self.path_start_index = np.zeros(agent_num,dtype=np.int),np.zeros(agent_num,dtype=np.int)

        self.step = steps
    
    def store(self , obs, image, action, reward, v_function, logp_value, agent_index=0 ):
        assert self.step > self.agent_step[agent_index]
        self.obs_buf[agent_index][self.agent_step[agent_index]] = obs
        self.img_buf[agent_index][self.agent_step[agent_index]] = image
        self.action_buf[agent_index][self.agent_step[agent_index]] = action
        self.reward_buf[agent_index][self.agent_step[agent_index]] = reward
        
        self.v_function_buf[agent_index][self.agent_step[agent_index]] = v_function
        self.logp_buf[agent_index][self.agent_step[agent_index]] =  logp_value
        self.agent_step[agent_index] += 1
        

    ###################
    #          GAE
    ##################
    def compute_advantage_value(self, last_v_value=0,agent_index=0):
        
        path = slice( self.path_start_index[agent_index] , self.agent_step[agent_index] )
      
        rewards = np.append(self.reward_buf[agent_index][path], last_v_value)
        v_value = np.append(self.v_function_buf[agent_index][path], last_v_value)

        delta = rewards[:-1] + self.gamma * v_value[1:] - v_value[:-1]

       
        self.advantage_buf[agent_index][path] = discount_sum(delta, self.gamma * self.lamda)
        
        ########################
        # compute the discount reward
        ########################
        self.discount_reward_buf[agent_index][path] =  discount_sum(rewards, self.gamma)[:-1]

        self.path_start_index[agent_index] = self.agent_step[agent_index]
        
        



    def get_buff(self, agent_index = 0):
        count = self.agent_step[agent_index]
        self.agent_step[agent_index], self.path_start_index[agent_index] = 0 , 0

        ##################################
        ### normaliza  the advantage_value
        ##################################

        temp = self.advantage_buf.flatten()[np.flatnonzero(self.advantage_buf)]
        advantage_mean = np.mean(temp)
        advantage_std = np.std(temp)
        notmalization_advantage_value_buf = (self.advantage_buf[agent_index] - advantage_mean) / advantage_std

        return [
                 self.obs_buf[agent_index][0:count], self.img_buf[agent_index][0:count] ,self.action_buf[agent_index][0:count],
                 notmalization_advantage_value_buf[0:count],self.discount_reward_buf[agent_index][0:count],
                 self.logp_buf[agent_index][0:count]
               ]



class PPO(nn.Module):
    def __init__(self, actor_critic,  agent_number=0 , resume_model=False ,seed=0, step=3000,gamma=0.99, clip_ratio=0.2,
         v_function_lr=1e-3, p_function_lr=3e-4 , lamda=0.97, targer_kl=0.01, train_pi_iter_num=80,train_v_iters_num=80):
         super(PPO,self).__init__()

         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

         self.train_pi_iters_num = train_pi_iter_num
         self.train_v_iters_num = train_v_iters_num
         self.targer_kl = targer_kl
         self.agent_num = agent_number
         self.normalization = StandardScaler()

         torch.manual_seed(seed)
         np.random.seed(seed)

         action_dim = config["action_dim"]
         state_dim = config["state_dim"]
         img_size = config["img_size"]
        ##############################
        ##  the compute graph
        #############################
         self.actor_critic = actor_critic
         self.actor_critic.to(self.device)

         self.clip_ratio = clip_ratio

         self.buf = PPOBuffer(state_dim,img_size,action_dim,step,gamma,lamda,agent_number)
         self.step = step

         if resume_model == True:
             reload_torch_model(path,self.actor_critic, itr=350)

         self.pi_optimization = Adam(self.actor_critic.p_function.parameters(), lr=p_function_lr)
         self.v_optimization = Adam(self.actor_critic.v_function.parameters(), lr=v_function_lr)

    ################
    # policy loss
    ###############
    def computer_loss_actor_pi(self, data):
        obs, image, action , advantage, log_p_old = data["obs"], data["image"], data["action"], data["advantage"],data["log_p_old"]
        pi , log_p = self.actor_critic.p_function(obs, image, action)
        

        ratio = torch.exp(log_p - log_p_old) #求log之后,把除法换成减法
        clip_a = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi_clip = -(torch.min(ratio * advantage, clip_a)).mean()

        kl = (log_p_old - log_p).mean().item() 
        
        return loss_pi_clip , kl

    ########################
    # v_function loss
    ########################
    def computer_loss_critic_v(self, data):
        obs , image, discount_reward = data["obs"], data["image"] , data["discount_reward"]
        return ((self.actor_critic.v_function(obs,image) - discount_reward)** 2).mean()



    def get_all_buff(self):
        obs_buff, image_buff, action_buff, normation_advantage_value_buff , discount_reward_buff, logp_value_buff = self.buf.get_buff(0)
        for i in range(1, self.agent_num):
            obs, image, action, normation_advantage_value , discount_reward, logp_value = self.buf.get_buff(i)
            obs_buff = np.append(obs_buff, obs, axis=0)
            image_buff = np.append(image_buff, image,axis=0)
            
            action_buff = np.append(action_buff, action, axis=0)
            normation_advantage_value_buff = np.append(normation_advantage_value_buff, normation_advantage_value, axis=0)
            discount_reward_buff = np.append(discount_reward_buff, discount_reward, axis=0)
            logp_value_buff = np.append(logp_value_buff, logp_value, axis=0)

        return obs_buff, image_buff, action_buff, normation_advantage_value_buff, discount_reward_buff, logp_value_buff

            



    def update(self):
        obs_buff, image_buff, action_buff, normation_advantage_value_buff, discount_reward_buff, logp_value_buff = self.get_all_buff()

        data = dict(obs=obs_buff, image=image_buff, action=action_buff, discount_reward = discount_reward_buff, advantage=normation_advantage_value_buff, log_p_old=logp_value_buff)

        input = {k : torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in list(data.items())}

        pi_loss_old, kl = self.computer_loss_actor_pi(input)

        pi_loss_old = pi_loss_old.item()

        v_function = self.computer_loss_critic_v(input).item()
        
        ######################################
        #train policy with policy gradient descent
        ######################################
        for i in range(self.train_pi_iters_num):
            self.pi_optimization.zero_grad()
            loss_pi , kl = self.computer_loss_actor_pi(input)
            kl = np.average(kl)

            loss_pi.backward()
            self.pi_optimization.step()

            if kl > 1.5 * self.targer_kl:
                print("reaching the max KL, early stop this %d episode"%i)
                break
        ###########################
        #train the V netwark
        for i in range(self.train_v_iters_num):
            self.v_optimization.zero_grad()
            loss_v = self.computer_loss_critic_v(input)
            loss_v.backward()
            self.v_optimization.step()

        return True



    def choose_action(self,  observations,  images):
        #obs = self.normalization.fit_transform(obs) #标准化

        observations = self.normalization.fit_transform(observations)
        
        action, v_function_vale , log_P = self.actor_critic.step(
           torch.as_tensor(observations, dtype=torch.float32,device=self.device),
           torch.as_tensor(images[np.newaxis, :], dtype=torch.float32,device=self.device),
        )#np.newaxis # 更改前面的维度,比如(4,4) 变成(1,4,4)

        ###########################
        # 可以在这里进行action masking,
        ##########################
        
        return action, v_function_vale , log_P 
    
    #############################
    ###   store the buffer
    ############################
    def buff_store(self, obs, image, action, reward, v_function, logp_value, agent_index):
        self.buf.store(obs, image, action, reward, v_function, logp_value, agent_index)
        
       

    def compute_advantage_value(self, last_value, agent_index):
        self.buf.compute_advantage_value(last_value, agent_index)

    def save_train_model(self, itr=-1):
        save_torch_model(path , self.actor_critic,itr=itr)
        
