
# Navigation

---

You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!

### 1. Start the Environment

Run the next code cell to install a few packages.  This line will take a few minutes to run!


```python
!pip -q install ./python
```

    [31mtensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible.[0m
    [31mipython 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 3.0.5 which is incompatible.[0m


The environment is already saved in the Workspace and can be accessed at the file path provided below.  Please run the next code cell without making any changes.


```python
from unityagents import UnityEnvironment
import numpy as np

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
```

    Number of agents: 1
    Number of actions: 4
    States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
      1.          0.          0.0748472   0.          1.          0.          0.
      0.25755     1.          0.          0.          0.          0.74177343
      0.          1.          0.          0.          0.25854847  0.          0.
      1.          0.          0.09355672  0.          1.          0.          0.
      0.31969345  0.          0.        ]
    States have length: 37


### 3. Take Random Actions in the Environment

In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.

Note that **in this coding environment, you will not be able to watch the agent while it is training**, and you should set `train_mode=True` to restart the environment.


```python
env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))
```

    Score: 1.0


When finished, you can close the environment.

### 4. It's Your Turn!

Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
- When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
```python
env_info = env.reset(train_mode=True)[brain_name]
```
- To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
- In this coding environment, you will not be able to watch the agent while it is training.  However, **_after training the agent_**, you can download the saved model weights to watch the agent on your own machine! 


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#Neural Network to calculate translation from state to action
class QNetwork(nn.Module):
    
    def __init__(self,state_size,action_size,seed):
        super(QNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,action_size)
    
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
```


```python
import numpy as np
import random

from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

#ALL Constant definitions
BUFFER_SIZE=1024
BATCH_SIZE=64
GAMMA=0.99
TAU=5e-4
LR=1e-4
UPDATE_EVERY=4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self,state_size,action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.qnetwork_target = QNetwork(state_size,action_size, seed).to(device)
        self.qnetwork_local = QNetwork(state_size,action_size, seed).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(),lr= LR)
        self.memory = ReplayBuffer(self.state_size,self.action_size,BUFFER_SIZE,BATCH_SIZE,seed)
        self.t_step = 0
    
    def step(self,state,action,reward,next_state,done):
        """
        Thif function adds experience into Memory buffer and run learning algorithm 
        if it has sufficient number of experiences
        """
        #Fill the memory buffer
        self.memory.add(state,action,reward,next_state,done)
      
        #Run learning at every time step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if(self.t_step == 0):
            if(len(self.memory) > BATCH_SIZE):
                experiences = self.memory.sample()
                self.learn(experiences,GAMMA)
              
    def act(self,state,eps = 0.):
        """This function basically returns next optimal action based on 
           state provided by using epsilon greedy policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
      
        if(random.random() > eps):
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))      
      
    def learn(self,experiences,gamma):
        """
        Training of Neural Network Algorithm 
        """
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)     
        loss = F.mse_loss(Q_expected,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
      
    def soft_update(seld,qnetwork_local,qnetwork_target,tau):
        """
         Soft update model parameters.
         θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_params,local_params in zip(qnetwork_target.parameters(),qnetwork_local.parameters()):
            target_params.data.copy_(( 1- tau) * target_params.data + tau * local_params.data)

#MEMORY BUFFER CLASS
class ReplayBuffer():
    
    def __init__(self,state_size,action_size,buffer_size,batch_size,seed):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",field_names = ["state","action","reward","next_state","done"])
        self.random = random.seed(seed)
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states,actions,rewards,next_states,dones)
        
    def __len__(self):
        return len(self.memory)
   
```


```python
agent = Agent(state_size, action_size, seed=0)
def dqn(episodes = 2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start                    
    for i_episode in range(episodes):
        env_info = env.reset(train_mode = True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state)
            env_info= env.step(action)[brain_name]         # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state,action,reward,next_state,done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn() 
```

    Episode 0	Average Score: 0.00
    Episode 100	Average Score: -0.21
    Episode 200	Average Score: 0.173
    Episode 300	Average Score: -0.06
    Episode 400	Average Score: 0.071
    Episode 500	Average Score: 0.09
    Episode 600	Average Score: 0.08
    Episode 700	Average Score: 1.36
    Episode 800	Average Score: 2.91
    Episode 900	Average Score: 4.44
    Episode 1000	Average Score: 7.85
    Episode 1100	Average Score: 8.61
    Episode 1200	Average Score: 9.775
    Episode 1300	Average Score: 9.72
    Episode 1400	Average Score: 4.740
    Episode 1500	Average Score: 9.05
    Episode 1600	Average Score: 11.90
    Episode 1700	Average Score: 7.758
    Episode 1800	Average Score: 11.06
    Episode 1900	Average Score: 12.20
    Episode 1912	Average Score: 13.00
    Environment solved in 1812 episodes!	Average Score: 13.00



```python
import matplotlib.pyplot as plt
%matplotlib inline

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()   
```


![png](images/output_15_0.png)



```python
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]
score = 0
while True:
    action = agent.act(state)
    env_info= env.step(action)[brain_name]         # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    agent.step(state,action,reward,next_state,done)
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
print("Score: {}".format(score))
```

    Score: 13.0



```python
env.close()
```

### 5 Further Improvements planned

We have implemented DQN agent and demonstrated how to use it to solve an Unity environment. Further we can  implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN.
   


