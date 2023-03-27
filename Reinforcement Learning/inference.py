import gym
import warnings

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

warnings.filterwarnings(action='ignore')

env = gym.make("LunarLander-v2")
states = env.observation_space.shape[0]
actions = env.action_space.n

def build_dl(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    buffer = SequentialMemory(limit=250000, window_length=1)
    dqn = DQNAgent(model=model, memory=buffer, policy=policy, 
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

model = build_dl(states, actions)
dqn = build_agent(model, actions)
dqn.compile(optimizer=Adam(), metrics=['mae'])

dqn.load_weights('Lunar Lander DQN.h5f')

_ = dqn.test(env, nb_episodes=15, visualize=True)
env.close()
