import numpy as np
from numpy.core.records import array
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

###################################
# build the network of ppo, there are two actor
#
#
###############################
config = {
    "img_batch" : 4, # 主要针对critic网络,critic网络中image是全局的image,image的数量与agent数量相对应
}

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.covlayer1 = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=(1,1)),
            nn.ReLU(),#nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer2 = nn.Sequential(
            nn.Conv2d(32,64,3,stride=1,padding=(1,1)),
           
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer3 = nn.Sequential(
            nn.Conv2d(64,64,3,stride=1,padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.fc1 = nn.Linear(64 * 64 * 64,64)
        self.fc2 = nn.Linear(139,64)
        #self.fc2 = nn.Linear(79,64)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,2)
        
        log_std = -0.5 * np.ones(2, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        #将一个不可训练的类型Tensor转换成可以训练的类型parameter
        #并将这个parameter绑定到这个module里面(net.parameter()中就有
        #这个绑定的parameter，所以在参数优化的时候可以进行优化的
    

    
    def get_distribution(self, obs, image):
        image_net = self.covlayer1(image)
        image_net = self.covlayer2(image_net)
        image_net = self.covlayer3(image_net)
        
        image_net = image_net.reshape(-1,64 * 64 * 64)

        image_out = self.fc1(image_net)
        #print("OOOOOOOOOOOOOOOO")
        #obs = obs.reshape(-1,79)
        obs = obs.reshape(-1,139)
        obs = self.fc2(obs)
        

        _net = torch.cat((image_out,obs), dim=1)
        

        _net = F.relu(self.fc3(_net)) #F.ReLU则作为一个函数调用，看上去作为一个函数调用更方便更简洁
        _net = F.relu(self.fc4(_net))
        _net = torch.sigmoid(self.fc5(_net))

        std = torch.exp(self.log_std)
        
        return Normal(_net,std)

    
    
    def log_prob(self, pi , act):
        return pi.log_prob(act).sum(axis=-1)



    def forward(self , obs , image, act):
        ##maybe the input's tensor will be modified!
        log_p_a = None
        
        pi = self.get_distribution(obs, image)
        
        if act is not None:
            log_p_a = self.log_prob(pi , act)

        return pi , log_p_a



class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()

        self.covlayer1 = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=(1,1)), 
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer2 = nn.Sequential(
            nn.Conv2d(32,64,3,stride=1,padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer3 = nn.Sequential(
            nn.Conv2d(64,64,3,stride=1,padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )
       

        #self.lstm = nn.LSTM(64,20,2)

        self.fc1 = nn.Linear( 64 * 64 * 64 , 64)
        self.fc2 = nn.Linear(139, 64)
        #self.fc2 = nn.Linear(79,64)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,1)

    def forward(self, obs,image ):#(64, 64, 3, 5)
        
        image_net = self.covlayer1(image)
        image_net = self.covlayer2(image_net)
        image_net = self.covlayer3(image_net)
        
        #h0 = torch.randn(2,3,20)
        #c0 = torch.randn(2,3,20)
        #image_net = self.lstm(image_net,(h0,c0))
       
        image_net = image_net.reshape(-1,  64 * 64 * 64)
        
        image_out = self.fc1(image_net)
        
        obs = self.fc2(obs.reshape(-1,139))
        #obs = self.fc2(obs.reshape(-1,79))
        
        _net = torch.cat((image_out,obs),dim=1)
        _net = _net.reshape(-1,128)

        _net = F.relu(self.fc3(_net))
        _net = F.relu(self.fc4(_net))
        _net = self.fc5(_net)
        

        return _net



class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        
        self.p_function = Actor()

        self.v_function = Critic()
#
    def step(self,  observations,  images):
        
        with torch.no_grad():
            pi = self.p_function.get_distribution(observations , images[0][1:])
            
            action = torch.tanh(pi.sample()) #* 0.5
            
            logp_a = self.p_function.log_prob(pi, action)

            v_function = self.v_function(observations, images[0][1:]) #critic network has all observation of the scenario
            
        return action.cpu().numpy(), v_function.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, observations, images):
        return self.step( observations,  images)




'''
class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.covlayer1 = nn.Sequential(
            nn.Conv2d(1,64,3,stride=1,padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.covlayer3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1,padding=(1,1))
        )

        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,2)
        
        log_std = -0.5 * np.ones(2, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        #将一个不可训练的类型Tensor转换成可以训练的类型parameter
        #并将这个parameter绑定到这个module里面(net.parameter()中就有
        #这个绑定的parameter，所以在参数优化的时候可以进行优化的
    
    def get_distribution(self, obs, image):
        actor_net = self.covlayer1(obs)
        actor_net = self.covlayer2(actor_net)
        actor_net = self.covlayer3(actor_net)

        _net = self.fc1(nn.ReLU(actor_net))
        _net = self.fc2(_net)
        _net = self.fc3(_net)
        _net = self.fc4(_net)

        
        return Categorical(_net)
    

    def log_prob(self, pi , act):
        return pi.log_prob(act)

    def forward(self, obs, image ,act=None):
        ##maybe the input's temsor will be modified!
        

        log_p_a = None
        pi = self.get_distribution(obs,image)

        if act is not None:
            log_p_a = self.log_prob(pi, act)

        return pi , log_p_a

'''
