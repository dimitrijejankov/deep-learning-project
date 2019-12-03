import cv2
import gym
import numpy as np
from pynput import keyboard
from gym import spaces
from time import sleep


class TrainingWrapper(gym.ObservationWrapper):
    # health starts at
    player_1_hp = 176
    player_2_hp = 176

    # how many times have we restarted
    num_resets = 0

    def observation(self, observation):
        return TrainingWrapper.process(observation)

    def __init__(self, swap_players, env=None, integration=None):
        super(TrainingWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64))
        self.integration = integration
        self.swap_players = swap_players

    def reset(self, **kwargs):

        # on a reset we set the health back to 176
        self.player_1_hp = 176
        self.player_2_hp = 176

        # reset the environment
        observation = self.env.reset(**kwargs)

        # we restarted inc the number
        self.num_resets += 1

        return self.observation(observation)

    def step(self, action):

        # perform one step
        observation, reward, done, info = self.env.step(action)

        # check if we got the health down to 0 for any of them
        if info['health1'] == 0 or info['health2'] == 0:

            # reset them back
            self.player_1_hp = 176
            self.player_2_hp = 176

            # the reward is 0
            reward = 0

        else:

            # save player 1 reward
            player_2_reward = self.player_1_hp - info['health1']
            self.player_1_hp = info['health1']

            # save player 2 reward
            player_1_reward = self.player_2_hp - info['health2']
            self.player_2_hp = info['health2']

            # attack and defend
            reward = player_2_reward * 8 - player_1_reward if self.swap_players \
                else player_1_reward * 8 - player_2_reward

        if done:
            # reset them back
            self.player_1_hp = 176
            self.player_2_hp = 176
            reward = 0

        # return the observation
        return self.observation(observation), reward, done, info

    @staticmethod
    def process(img):
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, (64, 64))
        x_t = np.nan_to_num(x_t)
        #x_t = np.rollaxis(x_t, 2, 0)
        x_t = np.array([x_t])
        return x_t.astype(np.uint8)


class PlayerOneNetworkControllerWrapper(gym.ActionWrapper):
    mapping = {
        #   P           U, D, R, L, K
        0: [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Up
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Down
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Left
        3: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Left + A
        4: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + B
        5: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + A + B
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Right
        7: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # Right + A
        8: [0, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + B
        9: [1, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + A + B
        10: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        11: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B
        12: [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Down A
        13: [0, 0, 0, 0, 0, 1, 0, 0, 1],  # Down B
        14: [0, 0, 0, 0, 0, 1, 0, 1, 1],  # Down Left B
        15: [0, 0, 0, 0, 0, 1, 1, 0, 1],  # Down Right B
        16: [1, 0, 0, 0, 0, 1, 0, 1, 0],  # Down Left A
        17: [1, 0, 0, 0, 0, 1, 1, 0, 0],  # Down Right A
    }

    def __init__(self, env):
        super(PlayerOneNetworkControllerWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(18)

    def action(self, action):

        # first goes the mapping for the player one, then an empty array
        a = self.mapping.get(action).copy()
        a.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        return a

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0

    def reverse_action(self, action):

        # get the left and right action
        a1 = action[:len(action) // 2]
        return self._reverse_action(a1)


class PlayerTwoNetworkControllerWrapper(gym.ActionWrapper):
    mapping = {
        #   P           U, D, R, L, K
        0: [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Up
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Down
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Left
        3: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Left + A
        4: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + B
        5: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + A + B
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Right
        7: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # Right + A
        8: [0, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + B
        9: [1, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + A + B
        10: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        11: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B
        12: [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Down A
        13: [0, 0, 0, 0, 0, 1, 0, 0, 1],  # Down B
        14: [0, 0, 0, 0, 0, 1, 0, 1, 1],  # Down Left B
        15: [0, 0, 0, 0, 0, 1, 1, 0, 1],  # Down Right B
        16: [1, 0, 0, 0, 0, 1, 0, 1, 0],  # Down Left A
        17: [1, 0, 0, 0, 0, 1, 1, 0, 0],  # Down Right A
    }

    def __init__(self, env):
        super(PlayerTwoNetworkControllerWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(18)

    def action(self, action):

        # first goes the mapping for the player one, then for the network
        a = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        a.extend(self.mapping.get(action).copy())
        return a

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0

    def reverse_action(self, action):

        # get the left and right action
        a1 = action[:len(action) // 2]
        return self._reverse_action(a1)


class VersusAgentControllerWrapper(gym.ActionWrapper):
    mapping = {
        #   P           U, D, R, L, K
        0: [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Up
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Down
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Left
        3: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Left + A
        4: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + B
        5: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + A + B
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Right
        7: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # Right + A
        8: [0, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + B
        9: [1, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + A + B
        10: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        11: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B
        12: [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Down A
        13: [0, 0, 0, 0, 0, 1, 0, 0, 1],  # Down B
        14: [0, 0, 0, 0, 0, 1, 0, 1, 1],  # Down Left B
        15: [0, 0, 0, 0, 0, 1, 1, 0, 1],  # Down Right B
        16: [1, 0, 0, 0, 0, 1, 0, 1, 0],  # Down Left A
        17: [1, 0, 0, 0, 0, 1, 1, 0, 0],  # Down Right A
    }

    def __init__(self, env, agent, swap_players):
        super(VersusAgentControllerWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(18)
        self.agent = agent
        self._observation = None
        self.swap_players = swap_players

    def step(self, action):
        self._observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(self._observation), reward, done, info

    def action(self, a1):

        # the action the first player is taking
        a1 = self.mapping.get(a1)

        # the action second player is taking
        a2 = self.mapping.get(self.agent.forward(self._observation)) if self._observation is not None \
            else self.mapping.get(0)

        # if donatello is fighting leonardo
        if not self.swap_players:
            a = a1.copy()
            a.extend(a2)
            return a

        # if leonardo is fighting donatello
        else:
            a = a2.copy()
            a.extend(a1)
            return a

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0

    def reverse_action(self, action):

        # get the left and right action
        a1 = action[:len(action) // 2]
        a2 = action[len(action) // 2:]

        return [self._reverse_action(a1), self._reverse_action(a2)]


up = False
down = False
left = False
right = False
a = False
s = False


def on_press(key):
    try:
        if key.char == 'a' or key.char == 'A':
            global a
            a = True
        elif key.char == 's' or key.char == 'S':
            global s
            s = True
    except AttributeError:
        if key == keyboard.Key.up:
            global up
            up = True
        elif key == keyboard.Key.down:
            global down
            down = True
        elif key == keyboard.Key.left:
            global left
            left = True
        elif key == keyboard.Key.right:
            global right
            right = True


def on_release(key):
    try:
        if key.char == 'a' or key.char == 'A':
            global a
            a = False
        elif key.char == 's' or key.char == 'S':
            global s
            s = False
    except AttributeError:
        if key == keyboard.Key.up:
            global up
            up = False
        elif key == keyboard.Key.down:
            global down
            down = False
        elif key == keyboard.Key.left:
            global left
            left = False
        elif key == keyboard.Key.right:
            global right
            right = False


class UserControllerWrapper(gym.ActionWrapper):
    mapping = {
        #   P           U, D, R, L, K
        0: [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Up
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Down
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Left
        3: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # Left + A
        4: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + B
        5: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # Left + A + B
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Right
        7: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # Right + A
        8: [0, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + B
        9: [1, 0, 0, 0, 0, 0, 1, 0, 1],  # Right + A + B
        10: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # A
        11: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B
        12: [1, 0, 0, 0, 0, 1, 0, 0, 0],  # Down A
        13: [0, 0, 0, 0, 0, 1, 0, 0, 1],  # Down B
        14: [0, 0, 0, 0, 0, 1, 0, 1, 1],  # Down Left B
        15: [0, 0, 0, 0, 0, 1, 1, 0, 1],  # Down Right B
        16: [1, 0, 0, 0, 0, 1, 0, 1, 0],  # Down Left A
        17: [1, 0, 0, 0, 0, 1, 1, 0, 0],  # Down Right A
    }

    def __init__(self, env, should_swap):
        super(UserControllerWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(18)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        self.should_swap = should_swap

    def action(self, action):

        global a
        global s
        global up
        global down
        global left
        global right

        # get the action from both players
        a2 = [1 if a else 0,
              0,
              0,
              0,
              1 if up else 0,
              1 if down else 0,
              1 if left else 0,
              1 if right else 0,
              1 if s else 0]

        if not self.should_swap:
            a = self.mapping.get(action).copy()
            a.extend(a2)
            return a
        else:
            a = a2.copy()
            a.extend(self.mapping.get(action))
            return a

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0

    def reverse_action(self, action):

        # get the left and right action
        a1 = action[:len(action) // 2]
        a2 = action[len(action) // 2:]

        return [self._reverse_action(a1), self._reverse_action(a2)]


class UserDataControllerWrapper(UserControllerWrapper):

    def __init__(self, env, should_swap):
        super(UserDataControllerWrapper, self).__init__(env, should_swap)

    observations = []
    actions = []

    step_num = 0

    def step(self, action):
        sleep(0.005)
        observation, reward, done, info = self.env.step(self.action(action))
        if self.step_num % 20 == 0:
            self.observations.append(observation)
            self.actions.append(action)
        self.step_num += 1
        return observation, reward, done, info
