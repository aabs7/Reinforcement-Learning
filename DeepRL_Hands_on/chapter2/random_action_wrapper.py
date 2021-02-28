import gym
from typing import TypeVar
import random

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self,env,epsilon = 0.1):
        super(RandomActionWrapper,self).__init__(env)
        self.epsilon = epsilon

# Here, we initialized our wrapper by calling a parents' __init__
# method and saving epsilon

    def action(self,action:Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()

        return action

if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    obs = env.reset()
    total_reward = 0

    while True:
        obs,reward,done,_ = env.step(0)
        total_reward += reward
        if done:
            break

    print("Reward got: %.2f" % total_reward)