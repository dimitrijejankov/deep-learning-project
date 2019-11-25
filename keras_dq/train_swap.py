import numpy as np
import retro
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from keras_dq.wrappers import VersusAgentControllerWrapper, TrainingWrapper
from keras_dq.model import create_model

# each player can make exactly 18 actions
num_actions = 18

# disable this for some reason it is necessary
tf.compat.v1.disable_eager_execution()

# add the custom path
retro.data.path("./")

# add the custom path
integration = retro.data.Integrations.CUSTOM_ONLY
integration.add_custom_path("roms")

# we start by training
main_env = retro.make(game='TeenageMutantNinjaTurtles-Nes',
                      state="DonVSLeo",
                      inttype=integration,
                      players=2)

# make two models
player_1 = create_model(name="player_1", action_space_n=num_actions)
player_2 = create_model(name="player_2", action_space_n=num_actions)

# load the models
player_1.load_weights("/home/dimitrije/git/deep-learning-project/keras_dq/player1_don.h5f")
player_2.load_weights("/home/dimitrije/git/deep-learning-project/keras_dq/player2_leo.h5f")

# how many rounds we need to play
max_rounds = 100

# go through a bunch of rounds of smaller 20 rounds where we train one model and fix the other
for r in range(max_rounds):

    # print what player is being trained
    print("Swapping players : player %s is being trained!" % (r % 2 + 1))

    # are we training donatello or leonardo
    shouldSwap = r % 2 != 0

    # the player we are training
    main_player = player_2 if shouldSwap else player_1

    # the player we are fixing
    fixed_player = player_1 if shouldSwap else player_2

    # apply action space wrapper
    env = TrainingWrapper(shouldSwap, main_env, integration)

    # apply observation space wrapper to reduce input size
    env = VersusAgentControllerWrapper(env, fixed_player, shouldSwap)

    # fit the main player for
    main_player.fit(env,
                    action_repetition=20,
                    nb_steps=4400,
                    nb_max_episode_steps=220,
                    visualize=True,
                    verbose=2)

    # save the weights
    if not shouldSwap:
        player_1.save_weights('player1_don_competitive.h5f', overwrite=True)
    else:
        player_2.save_weights('player2_leo_competitive.h5f', overwrite=True)
