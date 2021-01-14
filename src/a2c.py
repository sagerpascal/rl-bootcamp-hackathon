import os
import time
from collections import deque

import numpy as np
import tensorflow as tf

from ppo2 import explained_variance, mean_or_nan, swap_and_flatten
from utils import loghandler
from utils.distributions import CategoricalPdType

"""
Implementation of Advantage Actor Critic (A2C):

    Advantage Actor Critic, is a synchronous version of the A3C policy gradient method. As an alternative to the 
    asynchronous implementation of A3C, A2C is a synchronous, deterministic implementation that waits for each actor 
    to finish its segment of experience before updating, averaging over all of the actors. This more effectively uses 
    GPUs due to larger batch sizes.

----------------------------------------------

Created:
    14.01.2021, Pascal Sager <sage@zhaw.ch>

Paper:
    Asynchronous Methods for Deep Reinforcement Learning (https://arxiv.org/abs/1602.01783)

Code-Sources:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    https://github.com/hill-a/stable-baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
    https://github.com/openai/baselines

"""


class InverseTimeDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom Learning rate scheduler: Inverse linear time decay
    initial_learning_rate * (1 - step / number_of_updates)
    """

    def __init__(self, init_lr, number_of_updates):
        super().__init__()
        self.init_lr = init_lr
        self.number_of_updates = number_of_updates

    def __call__(self, step):
        """
        Returns the new learning rate: initial_learning_rate * (1 - step / number_of_updates)
        :param step: The current step in the environment
        :return: the new learning rate
        """
        with tf.name_scope('inverse_time_decay'):
            init_lr = tf.convert_to_tensor(self.init_lr)
            number_of_updates_t = tf.convert_to_tensor(self.number_of_updates, dtype=init_lr.dtype)
            step_t = tf.cast(step, init_lr.dtype)
            return init_lr * (1.0 - step_t / number_of_updates_t)

    def get_config(self):
        """
        Method which stores the tf config (must be overwritten)
        """
        return {
            "init_lr": self.init_lr,
            "number_of_updates": self.number_of_updates,
            "name": 'inverse_time_decay'
        }


class PolicyAndValueNetwork(tf.Module):

    def __init__(self, ac_space, ob_space):
        super().__init__()
        input_layer = tf.keras.Input(shape=ob_space.shape)
        layers = input_layer
        for i in range(2):
            layers = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(layers)

        self.policy_network = tf.keras.Model(inputs=[input_layer], outputs=[layers])
        self.policy_distribution = CategoricalPdType(self.policy_network.output_shape, ac_space.n, init_scale=0.01)

        print(self.policy_network.summary())

        with tf.name_scope('value_function'):
            layer = tf.keras.layers.Dense(units=1, bias_initializer=tf.keras.initializers.Constant(0.01))
            layer.build(self.policy_network.output_shape)
        self.value_fc = layer

    @tf.function
    def step(self, observation):
        """
        Calcualte action for a given observation
        """
        latent = self.policy_network(observation)
        pd, _ = self.policy_distribution.pdfromlatent(latent)
        action = pd.sample()
        vf = tf.squeeze(self.value_fc(latent), axis=1)
        return action, vf

    @tf.function
    def value(self, observation):
        """
        Compute value given the observation
        """
        latent = self.policy_network(observation)
        result = tf.squeeze(self.value_fc(latent), axis=1)
        return result


class ActorCriticModel(tf.keras.Model):

    def __init__(self, action_space, observation_space, number_of_updates, lr):
        super().__init__()
        self.train_model = PolicyAndValueNetwork(action_space, observation_space)
        lr_scheduler = InverseTimeDecay(init_lr=lr, number_of_updates=number_of_updates)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_scheduler, rho=0.99, epsilon=1e-5)

        self.entropy_coefficient = 0.01
        self.value_function_coefficient = 0.5
        self.max_grad_norm = 0.5
        self.step = self.train_model.step
        self.value = self.train_model.value

    @tf.function
    def train(self, obs, rewards, actions, values):
        """
        Train the network: First, calculate the policy distribution based on the current observation (with the policy
        network). From this distribution, the policy loss and the policy entropy can be calculated. Then estimate the
        value with the same network.
        """
        advantage = rewards - values
        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            policy_distribution, _ = self.train_model.policy_distribution.pdfromlatent(policy_latent)
            neg_log_probability = policy_distribution.neglogp(actions)
            policy_entropy = tf.reduce_mean(policy_distribution.entropy())

            # compute the estimated value given the observation (the critic)
            value_predicted = self.train_model.value(obs)

            # compute the overall loss
            value_loss = tf.reduce_mean(tf.square(value_predicted - rewards))
            policy_loss = tf.reduce_mean(advantage * neg_log_probability)
            loss = policy_loss - policy_entropy * self.entropy_coefficient + value_loss * self.value_function_coefficient

        # Update the gradients
        variables = tape.watched_variables()
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        gradients_variables = list(zip(gradients, variables))
        self.optimizer.apply_gradients(gradients_variables)

        return policy_loss, value_loss, policy_entropy


def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done
    """
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


class MinibatchExecuter:

    def __init__(self, environment, model, number_of_steps, gamma):
        self.gamma = gamma
        self.environment = environment
        self.model = model
        self.number_of_steps = number_of_steps
        self.number_of_envs = environment.num_envs if hasattr(environment, 'num_envs') else 1
        self.observations = np.zeros((self.number_of_envs,) + environment.observation_space.shape,
                                     dtype=environment.observation_space.dtype.name)
        self.observations[:] = environment.reset()
        self.dones = [False for _ in range(self.number_of_envs)]

    def run(self):
        observations_minibatch, rewards_minibatch, actions_minibatch, values_minibatch, dones_minibatch = [], [], [], [], []
        episode_infos = []

        for _ in range(self.number_of_steps):
            # Get action and value for a given observation
            observations_t = tf.constant(self.observations)
            actions, values, = self.model.step(observations_t)
            actions = actions._numpy()
            observations_minibatch.append(self.observations.copy())
            actions_minibatch.append(actions)
            values_minibatch.append(values._numpy())
            dones_minibatch.append(self.dones)

            # Take actions in env and look the results
            self.observations[:], rewards, self.dones, logs = self.environment.step(actions)
            for log in logs:
                episode_log = log.get('episode')
                if episode_log:
                    episode_infos.append(episode_log)
            rewards_minibatch.append(rewards)

        dones_minibatch.append(self.dones)

        # Batch of steps to batch of rollouts
        observations_minibatch = swap_and_flatten(np.asarray(observations_minibatch, dtype=self.observations.dtype))
        rewards_minibatch = np.asarray(rewards_minibatch, dtype=np.float32).swapaxes(1, 0)
        actions_minibatch = swap_and_flatten(np.asarray(actions_minibatch, dtype=actions.dtype))
        values_minibatch = np.asarray(values_minibatch, dtype=np.float32).swapaxes(1, 0)
        dones_minibatch = np.asarray(dones_minibatch, dtype=np.bool).swapaxes(1, 0)
        dones_minibatch = dones_minibatch[:, 1:]

        if self.gamma > 0.0:
            # Discount value function
            last_values = self.model.value(tf.constant(self.observations))._numpy().tolist()
            for i, (rewards, dones, value) in enumerate(zip(rewards_minibatch, dones_minibatch, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                rewards_minibatch[i] = rewards

        rewards_minibatch = rewards_minibatch.flatten()
        values_minibatch = values_minibatch.flatten()
        return observations_minibatch, rewards_minibatch, actions_minibatch, values_minibatch, episode_infos


def load_checkpoint(conf, model):
    if conf['load_path'] is not None and conf['load_path'] is not 'None':
        load_path = os.path.expanduser(conf['load_path'])
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)


def get_model(env, conf, number_of_updates):
    ob_space = env.observation_space
    ac_space = env.action_space
    return ActorCriticModel(action_space=ac_space, observation_space=ob_space, number_of_updates=number_of_updates,
                            lr=conf['lr'])


def train(env, conf):
    total_timesteps = int(10e7)
    logger = loghandler.LogHandler(conf)

    # Calculate the batch_size
    number_of_envs = env.num_envs
    assert env.num_envs == conf['num_env']
    number_of_batches = number_of_envs * conf['nsteps']
    number_of_updates = total_timesteps // number_of_batches

    model = get_model(env, conf, number_of_updates)
    load_checkpoint(conf, model)

    # Instantiate the runner object
    minibatch_executer = MinibatchExecuter(env, model, number_of_steps=conf['nsteps'], gamma=conf['gamma'])
    episode_infos = deque(maxlen=100)

    best_reward = -100000000000
    time_start = time.time()

    for update in range(1, number_of_updates + 1):
        logs = {}
        # Get mini batch of experiences
        observations, rewards, actions, values, epinfos = minibatch_executer.run()
        episode_infos.extend(epinfos)

        observations = tf.constant(observations)
        rewards = tf.constant(rewards)
        actions = tf.constant(actions)
        values = tf.constant(values)
        policy_loss, value_loss, policy_entropy = model.train(observations, rewards, actions, values)
        running_time_sec = time.time() - time_start

        fps = int((update * number_of_batches) / running_time_sec)
        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        ev = explained_variance(values, rewards)
        mean_reward = mean_or_nan([epinfo['r'] for epinfo in episode_infos])
        logs["number_of_updates"] = update
        logs["total_timesteps"] = update * number_of_batches
        logs["fps"] = fps
        logs["policy_entropy"] = float(policy_entropy)
        logs["value_loss"] = float(value_loss)
        logs["explained_variance"] = float(ev)
        logs["mean_episode_reward"] = mean_reward
        logs["mean_episode_length"] = mean_or_nan([epinfo['l'] for epinfo in episode_infos])
        logger.log(logs)

        if update > 15000 and mean_reward > best_reward:
            best_reward = mean_reward
            save_path = os.path.expanduser('best_models/a2c')
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
            manager.save()


def val(env, conf):
    observation = env.reset()

    if conf['load_path'] is None or conf['load_path'] is 'None':
        raise AttributeError("Load path must be defined to validate model!!!")

    total_timesteps = int(10e6)

    # Calculate the batch_size
    number_of_batches = conf['num_env'] * conf['nsteps']
    number_of_updates = total_timesteps // number_of_batches

    model = get_model(env, conf, number_of_updates)
    load_checkpoint(conf, model)

    episode_reward = np.zeros(env.num_envs)
    while True:
        actions, _ = model.step(observation)
        observation, reward, done, _ = env.step(actions.numpy())
        episode_reward += reward
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode reward = {}'.format(episode_reward[i]))
                episode_reward[i] = 0
