'''
Deep Deterministic policy Gradient (DDPG), Reinforcement Learning
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyper Parameters
EPISODES = 200
EP_STEPS = 200
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32                  
RENDER = False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.tensor(action_bound)
        # layer
        self.fc1 = nn.Linear(state_dim, 30)
        self.fc1.weight.data.normal_(0., 0.3)  
        # self.fc1.bias.data.fill_(0.1)
        self.out = nn.Linear(30, action_dim)
        self.out.weight.data.normal_(0., 0.3)
        # self.out.bias.data.fill_(0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x * self.action_bound
        
        return actions
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fcs = nn.Linear(state_dim, 30)
        self.fcs.weight.data.normal_(0, 0.3)
        self.fcs.bias.data.fill_(0.1)
        self.fca = nn.Linear(action_dim, 30)
        self.fca.weight.data.normal_(0, 0.3)
        self.fca.bias.data.fill_(0.1)
        self.out = nn.Linear(30, 1)
        
    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        
        return actions_value
    
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound):        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound)
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim*2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0 # serves as updating the memory data
        
        # Create the 4 network objects
        self.actor_eval = Actor(state_dim, action_dim, action_bound)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        self.critic_eval = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr = LR_CRITIC)
        
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()
        
    def store_transition(self, s, a, r, s_): 
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        
    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        a = self.actor_eval(s)[0]
        
        return a.detach().numpy()
    
    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)') 
        
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        
        batch_s = torch.FloatTensor(batch_trans[:, :self.state_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.state_dim:self.state_dim + self.action_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.state_dim - 1: -self.state_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.state_dim:])
        
        # make actions and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()
        

