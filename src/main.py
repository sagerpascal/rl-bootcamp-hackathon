import argparse
import multiprocessing
import random

import gym
import numpy as np
import tensorflow as tf
import yaml

import a2c
import ddpg
import deepq
import ppo2
from env_wrapper.dummy_vec_env import DummyVecEnv
from env_wrapper.monitor import Monitor
from env_wrapper.subproc_vec_env import SubprocVecEnv

# Settings
ENV = 'LunarLander-v2'  # either LunarLander-v2 or LunarLanderContinuous-v2
ENV_TYPE = 'classic_control'
seed = 7

# Set seed
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_env():
    env = gym.make(ENV)

    if isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed)
    env = Monitor(env, 'pascals_logs.txt', allow_early_resets=True)

    return env


def make_vec_env(number_of_envs):
    """
    Create a wrapped, monitored Env
    """

    def make_env_func():
        return lambda: make_env()

    if number_of_envs > 1:
        return SubprocVecEnv([make_env_func() for _ in range(number_of_envs)])
    else:
        return DummyVecEnv([make_env_func()])


def build_env(conf):
    number_of_envs = conf['num_env'] or multiprocessing.cpu_count()
    env = make_vec_env(number_of_envs or 1)
    return env


def main(conf):
    env = build_env(conf)

    if conf['mode'] == 'train':
        if ENV == 'LunarLander-v2':
            if conf['algorithm'] == 'a2c':
                # a2c.train(env, conf)
                a2c.train(env, conf)
            elif conf['algorithm'] == 'deepq':
                deepq.train(env, conf)
            elif conf['algorithm'] == 'ppo2':
                ppo2.train(env, conf)
            else:
                raise NotImplementedError("Unknown algorithm {} for LunarLander-v2".format(conf['algorithm']))

        elif ENV == "LunarLanderContinuous-v2":
            if conf['algorithm'] == 'ddpg':
                ddpg.train(env, conf)
            else:
                raise NotImplementedError("Unknown algorithm {} for LunarLanderContinuous-v2".format(conf['algorithm']))

    elif conf['mode'] == 'val':
        if ENV == 'LunarLander-v2':
            if conf['algorithm'] == 'a2c':
                a2c.val(env, conf)
            elif conf['algorithm'] == 'deepq':
                deepq.val(env, conf)
            elif conf['algorithm'] == 'ppo2':
                ppo2.val(env, conf)
            else:
                raise NotImplementedError("Unknown algorithm {} for LunarLander-v2".format(conf['algorithm']))
        elif ENV == "LunarLanderContinuous-v2":
            if conf['algorithm'] == 'ddpg':
                ddpg.val(env, conf)
            else:
                raise NotImplementedError("Unknown algorithm {} for LunarLanderContinuous-v2".format(conf['algorithm']))

    env.close()


def get_config():
    conf_file = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate")
    parser.add_argument("--load_path")
    parser.add_argument("--gamma")
    args = parser.parse_args()

    args_dict = {
        'load_path': str(args.load_path),
        'lr': float(args.learning_rate),  # 7e-4
        'gamma': float(args.gamma),  # 0.99
    }

    return {**conf_file, **args_dict}


if __name__ == '__main__':
    main(get_config())
