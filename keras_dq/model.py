from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.callbacks import Callback
from keras.optimizers import Adam


class RewardLogger(Callback):

    def __init__(self):
        super(RewardLogger, self).__init__()
        self.total = 0
        self.num = 0

    def on_episode_end(self, step, logs=None):
        self.total += logs["episode_reward"]
        self.num += 1


def create_model(name, action_space_n):
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
    model.add(Dense(action_space_n, activation='linear'))

    # set the policy
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)

    # make the player 1 agent
    player = DQNAgent(model=model,
                      nb_actions=action_space_n,
                      enable_dueling_network=True,
                      enable_double_dqn=True,
                      memory=memory,
                      nb_steps_warmup=200,
                      target_model_update=1e-2,
                      policy=policy)

    player.compile(Adam(lr=1e-3), metrics=['mae'])
    return player
