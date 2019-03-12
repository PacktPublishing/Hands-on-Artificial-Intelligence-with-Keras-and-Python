import gym 

env = gym.make("MountainCar-v0") 
observation = env.reset() 

for _ in range(10000): 
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(observation, reward, done, info)

