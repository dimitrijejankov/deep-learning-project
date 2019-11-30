import numpy as np
import retro

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras_dq.model import create_model

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
env = UserControllerWrapper(env, False)

# 3. Apply observation space wrapper to reduce input size
env = TrainingWrapper(False, env, integration)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# make the model
# make the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", input_shape=(4, 64, 64),
                 data_format="channels_first"))
model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# set the policy
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

# make the player 1 agent
dqn = DQNAgent(model=model,
               nb_actions=env.action_space.n,
               enable_dueling_network=True,
               enable_double_dqn=True,
               memory=memory,
               nb_steps_warmup=200,
               target_model_update=1e-2,
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
print(model.summary())

dqn.load_weights('/home/dimitrije/git/deep-learning-project/trained-models/dqn/player1_don_competitive_4_reward_227.0.h5f')
#dqn.load_weights('/home/dimitrije/git/deep-learning-project/trained-models/player2_leo_competitive_5_reward_140.0.h5f')
#dqn.load_weights('/home/dimitrije/git/deep-learning-project/trained-models/player2_leo_competitive_3_reward_107.0.h5f')

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)