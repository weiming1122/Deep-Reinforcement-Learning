import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyper Parameters
BATCH_SIZE = 32                  # 样本数量 batch size
LR = 0.01                        # 学习率 learning rate
EPSILON = 0.9                    # greedy policy
GAMMA = 0.9                      # reward discount
TARGET_REPLACE_ITER = 100        # 目标网络更新频率 target network update frequency
MEMORY_CAPACITY = 2000           # 记忆库容量

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 30)        # layer 1
        self.fc1.weight.data.normal_(0, 0.1)      # 权重初始化（均值为0，方差为0.1的正态分布）
        self.out = nn.Linear(30, N_ACTIONS)       # layer 2
        self.out.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        
        return actions_value
    
class DQN(object):
    def __init__(self, N_STATES, N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)                  # 评估网络和目标网络
        self.learn_step_counter = 0                                    # for targer updating
        self.memory_counter = 0                                        # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))        # 初始化记忆库 initialize memory，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 使用Adam优化器
        self.loss_func = nn.MSELoss()             # 使用均方损失函数 (loss(xi, yi) = (xi-yi)^2)
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)    # add 1 dimension to input state x
        if np.random.uniform() < EPSILON:               # 生成一个在[0,1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)
            #print(torch.max(actions_value, 1))
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]
        else:                                  
            action = np.random.randint(0, self.N_ACTIONS)    #随机选择动作
        
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition         # replace the old memory with new memory
        self.memory_counter += 1
        
    def learn(self):
        # target network parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1
        
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        
        # the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()      
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()