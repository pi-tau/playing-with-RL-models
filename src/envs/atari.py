import gym
import numpy as np

from ..core import Environment, TimeStep

# MsPacman actions
# ================
# 0 : NOOP
# 1 : NORTH
# 2 : EAST
# 3 : WEST
# 4 : SOUTH
# 5 : NORTHEAST
# 6 : NORTHWEST
# 7 : SOUTHEAST
# 8 : SOUTHWEST


class AtariEnvironment(Environment):

    def __init__(self, name, frameskip=4, framestack=4,
                 obs_type='grayscale', ghosts=1):
        """
        name (str):
            Environment name in OpenGym convention.
        skipframes (int):
            Number of frames to skip after calling step() method.
        stackframes (int):
            Number of frames to stack for single observation.
        """
        self._gymenv = gym.make(
            name,
            frameskip=frameskip,
            repeat_action_probability=0.0,
            difficulty=0,
            mode=ghosts,
            obs_type=obs_type
        )
        self.framestack = framestack

    def step(self, action):
        observation, reward, done, info = self._gymenv.step(action)
        frames = [observation / 255.0]
        for _ in range(self.framestack - 1):
            frame, r, d, info = self._gymenv.step(0)
            frames.append(frame / 255.0)
            reward = max(reward, r)
            done = done or d
        observation = np.array(frames)
        return TimeStep(observation, reward, done, info)

    def reset(self):
        observation = self._gymenv.reset()
        frames = [observation / 255.0]
        for _ in range(self.framestack - 1):
            frame, _, _, _ = self._gymenv.step(0)
            frames.append(frame / 255.0)
        observation = np.array(frames)
        return TimeStep(observation, 0, False, {})

    def actions(self):
        return list(range(self._gymenv.action_space.n))

    def num_actions(self):
        return self._gymenv.action_space.n

    def shape(self):
        return (self.framestack,) + self._gymenv.observation_space.shape

    def close(self):
        self._gymenv.close()
