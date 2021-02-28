import gym
e = gym.make('CartPole-v0')

obs = e.reset()
print("Observation:")
print(obs)

print("\naction_space:")
print(e.action_space)

print("\nObservation space")
print(e.observation_space)

print("\nenvironment step")
print(e.step(0))

print("\naction_space sample")
print(e.action_space.sample())

print("\nObservation space sample")
print(e.observation_space.sample())
