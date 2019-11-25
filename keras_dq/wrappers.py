import cv2
import gym
import numpy as np
import keyboard
from gym import spaces


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

            # just focus on how much dmg we are doing
            reward = player_2_reward * 4 if self.swap_players else player_1_reward * 4

        if done:
            # reset them back
            self.player_1_hp = 176
            self.player_2_hp = 176
            reward = 0

        # return the observation
        return self.observation(observation), reward, done, info

    @staticmethod
    def process(img):
        # img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # x_t = np.reshape(x_t, (64, 64))
        x_t = np.nan_to_num(x_t)
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

    def __init__(self, env):
        super(UserControllerWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(18)

    def action(self, action):

        # get the action from both players
        a2 = [keyboard.is_pressed('a'),
              0,
              0,
              0,
              keyboard.is_pressed('y'),
              keyboard.is_pressed('h'),
              keyboard.is_pressed('j'),
              keyboard.is_pressed('g'),
              keyboard.is_pressed('s')]

        a = self.mapping.get(action[0]).copy()
        a.extend(a2)

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
