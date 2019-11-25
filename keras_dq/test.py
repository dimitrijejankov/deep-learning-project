import numpy as np
import retro

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

from keras_dq.wrappers import UserControllerWrapper, TrainingWrapper

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 1. Create gym environment
retro.data.path("./")

# add the custom path
integration = retro.data.Integrations.CUSTOM_ONLY
integration.add_custom_path("roms")

#env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="LeoVSDonCPU", inttype=integration, players=1)
env = retro.make(game='TeenageMutantNinjaTurtles-Nes', state="DonVSLeo", inttype=integration, players=2)

# 2. Apply action space wrapper
env = UserControllerWrapper(env)

# 3. Apply observation space wrapper to reduce input size
env = TrainingWrapper(env, integration)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", input_shape=(1, 64, 64), data_format="channels_first"))
model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               enable_dueling_network=True,
               memory=memory,
               nb_steps_warmup=1000,
               target_model_update=1e-2,
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
print(model.summary())

dqn.load_weights('/home/dimitrije/git/deep-learning-project/keras_dq/dql_player_1_vs_nobody.h5f')

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)