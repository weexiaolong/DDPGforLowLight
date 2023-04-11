import gym
from gym import spaces
import numpy as np
import cv2
import os


class LocalImageEnv(gym.Env):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(self.img_dir)
        self.action_space = spaces.Discrete(len(self.img_list))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.current_index = None
        self.current_observation = None

    def step(self, agent):

        img_path = os.path.join(self.img_dir, self.img_list[self.current_index])
        observation = cv2.imread(img_path)
        observation = cv2.resize(observation, (84, 84))
        action = agent.actor(observation)


        self.current_observation = observation
        reward = 1 if self.current_index == action else 0
        done = True if self.current_index == len(self.img_list) - 1 else False
        info = {}
        self.current_index = action
        return observation, reward, done, info

    def reset(self):
        self.current_index = 0
        img_path = os.path.join(self.img_dir, self.img_list[self.current_index])
        observation = cv2.imread(img_path)
        observation = cv2.resize(observation, (84, 84))
        self.current_observation = observation
        return observation

    def render(self, mode='human'):
        if self.current_observation is not None:
            cv2.imshow('image', self.current_observation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('No observation available')
