import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'MountainCar-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(999)
env.seed(999)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
model.compile(loss='mse', optimizer=Adam())
print(model.summary())

# Finally, we configure and compile our agent. 

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
               nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# You can always safely abort the training prematurely using Ctrl + C.
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weight.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=10, visualize=True)
