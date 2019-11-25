import numpy as np
import retro
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from keras_dq.wrappers import PlayerOneNetworkControllerWrapper, TrainingWrapper


# disable this for some reason it is necessary
tf.compat.v1.disable_eager_execution()

# add the custom path
retro.data.path("./")

# add the custom path
integration = retro.data.Integrations.CUSTOM_ONLY
integration.add_custom_path("roms")

# we start by training
env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="DonVSLeo", inttype=integration, players=2)

# this wraps the environment and shapes the input and reward so that we like it
env = TrainingWrapper(False, env, integration)

# this also wraps the environment it translates the output of the nn to the player 1 controls
env = PlayerOneNetworkControllerWrapper(env)

# make the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", input_shape=(64, 64, 3),
                 data_format="channels_last"))
model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# set the policy
policy = MaxBoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

# make the player 1 agent
player1 = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   enable_dueling_network=True,
                   enable_double_dqn=True,
                   memory=memory,
                   nb_steps_warmup=200,
                   target_model_update=1e-2,
                   policy=policy)

player1.compile(Adam(lr=1e-3), metrics=['mae'])
print(model.summary())

# do the training for at least 20 episodes 220 * 20 = 4400
player1.fit(env, action_repetition=20, nb_steps=4400, nb_max_episode_steps=220, visualize=True, verbose=2)

# After training is done, we save the final weights.
player1.save_weights('player1_don.h5f', overwrite=True)
