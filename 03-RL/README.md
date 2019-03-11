# Section 3: Reinforcement learning, Self-learning AI game agent

This section has two parts: 
* **Part-I**: Here we will look at standard RL benchmarking framework. We will fire-up a simple game and develop a DQN to solve a particular scenario.
* **Part-II**: In this project we play Google Chrome's Dino-run game using Reinforcement Learning. The RL algorithm is based on the [Deep Q-Learning algorithm](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and is implemented from scratch in TensorFlow and Keras. The project is inspired from the tensorflow implementation version found [here](https://vdutor.github.io/blog/2018/05/07/TF-rex.html). This kind of project will give good intuition on how to apply RL on different scenarios apart from OpenAI Gym environment.

## **Part-I: OpenAI Gym environment**
---

Gym is a toolkit for benchmarking RL algorithms. A sample environment is presented below in `mountaincar-env.py`

```
import gym 

env = gym.make("MountainCar-v0") 
observation = env.reset() 

for _ in range(1000): 
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(observation, reward, done, info)
```

To run the sample environment type in

```sh
$ python mountaincar-env.py
```

## **Part-II: DinoRun game AI agent and replay**
---

 * ### Installation

Tested on Ubuntu 16.04 LTS. Start by cloning the repository and navigate to `03-RL` folder
```sh
$ git clone https://github.com/PacktPublishing/Hands-on-Artificial-Intelligence-with-Keras-and-Python.git
```

If you have not followed the steps in `01-Intro` to setup your environment already it would be good to have a look at that section again.


 * ### Running the Javascript DinoRun game

A simple HTTP Server module is used to deploy the game and play it on the browser. Open a new terminal and navigate to `game` folder and run a http server as follows
```sh
$ cd ~/Hands-on-Artificial-Intelligence-with-Keras-and-Python/03-RL/DinoRunGame/game/
$ python2 -m SimpleHTTPServer 8000
```
The game is now accessable on your localhost. Open your browser (Chrome) and type in `127.0.0.1:8000`.


 * ### DinoRun game AI agent with a pre-trained model

First, all the commandline arguments can be retrieved with
```sh
$ cd ~/Hands-on-Artificial-Intelligence-with-Keras-and-Python/03-RL/DinoRunGame/rl-agent
$ python main.py --help
```
Quickly check if the installation was successful by playing with a pretrained Q-learner.
```sh
$ python main.py --notraining --logdir ../trained-model
```
This command will restore the pretrained model, stored in `../trained-model` and play the Dino-run game.

IMPORTANT: The browser needs to connect with the python side. Therefore, refresh the browser after firing `python main.py --notraining --logdir ../trained-model`.


 * ### Training a DinoRun game AI agent with Reinforcement Learning

Training a new model can be done as follows
```sh
$ python main.py --logdir <logs>
```
Again, the browser needs to be refreshed to start the process. The directory passed as `logdir` argument will be used to store intermediate checkpoints and tensorboard information.

While training, a different terminal can be opened to launch the tensorboard
```sh
$ tensorboard --logdir <logs>
```
The tensorboards will be visible on `http://127.0.0.1:6006/`.

