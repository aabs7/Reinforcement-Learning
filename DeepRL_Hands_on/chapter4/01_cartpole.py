import gym
import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple



# Our model's core is one hidden layer NN, with ReLU and 128 hidden neurons
HIDDEN_SIZE = 128
# The count of episode we play on every iteration (16)
BATCH_SIZE = 16
# Percentile of the episodes' total reward that we use for "elite" episode filtering
PERCENTILE = 70
# We will take the 70th percentile, which means we will leave the top 30% of episodes sorted by reward. 


class Net(nn.Module):
    def __init__(self,obs_size,hidden_size,n_actions):
        super(Net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions)
        )
    
    def forward(self,x):
        return self.net(x)

# In above NN, Observation as input vector, output as actions.

Episode = namedtuple(
    'Episode', field_names = ['reward','steps']
)
EpisodeStep = namedtuple(
    'EpisodeStep', field_names = ['observation','action']
)

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim = 1)

    while True:
        obs_v = torch.FloatTensor([obs])
        action_probs_v = sm(net(obs_v))
        # see this step below 
        action_probs = action_probs_v.data.numpy()[0]

        action = np.random.choice(len(action_probs),p = action_probs)
        next_obs, reward, is_done , _  = env.step(action)

        episode_reward += reward
        step = EpisodeStep(observation = obs,action = action)
        episode_steps.append(step)

        if is_done:
            e = Episode(reward = episode_reward,steps = episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []