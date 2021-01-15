import tensorflow as tf
import numpy as np

"""
Implementation of Deep Deterministic Policy Gradients (A2C):

    Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It 
    uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

----------------------------------------------

Created:
    15.01.2021, Pascal Sager <sage@zhaw.ch>

Paper:
    Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)

Code-Sources:
    https://github.com/slowbull/DDPG
    https://github.com/hill-a/stable-baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
    https://github.com/openai/baselines

"""


class NumpyBuffer:

    def __init__(self, size, shape=(1,)):
        self.size = size
        self.items = np.zeros((size,) + shape).astype('float32')
        self.head = 0
        self.count = 0

    def __getitem__(self, item):
        return self.items[(self.head + item) % self.size]

    def append(self, v):
        if self.count < self.size:
            self.count += 1
        elif self.count == self.size:
            # Replace first item
            self.head = (self.head + 1) % self.size
        self.items[(self.head + self.count - 1) % self.size] = v


class DataStorage:

    def __init__(self, size):
        self.size = size
        self.observations = NumpyBuffer(size)
        self.actions = NumpyBuffer(size)
        self.rewards = NumpyBuffer(size)
        self.observations_next = NumpyBuffer(size)

    def add_data(self, observation, action, reward, next_observation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations_next.append(next_observation)

    def get_batch(self, batch_size):
        batch_indexes = np.random.randint(len(self.observations), size=batch_size)
        observation_batch = self.observations.get_batch(batch_indexes)
        action_batch = self.actions.get_batch(batch_indexes)
        reward_batch = self.rewards.get_batch(batch_indexes)
        observation_next_batch = self.observations_next.get_batch(batch_indexes)
        return observation_batch, action_batch, reward_batch, observation_next_batch


def create_network(input_shape):
    # TODO: try different parameters
    input_layer = tf.keras.Input(shape=input_shape)
    layers = input_layer
    for i in range(2):
        layers = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(layers)
    network = tf.keras.Model(inputs=[input_layer], outputs=[layers])
    return network


class Actor(tf.keras.Model):

    def __init__(self, observation_shape):
        number_of_actions = 2
        self.network = create_network(observation_shape)
        self.output_layer = tf.keras.layers.Dense(units=number_of_actions, activation=tf.keras.activations.relu)(self.network.outputs[0])

    @tf.function
    def call(self, obs):
        return self.output_layer(self.network(obs))


class Critic(tf.keras.Model):

    def __init__(self, observation_shape):
        number_of_actions = 2
        self.network = create_network((observation_shape[0] + number_of_actions,))
        self.output_layer = tf.keras.layers.Dense(units=self.number_of_actions, activation=tf.keras.activations.relu)(self.network.outputs[0])

    @tf.function
    def call(self, observations, actions):
        observations_actions = tf.concat([observations, actions], axis=-1)
        observations_actions = self.network_builder(observations_actions)
        return self.output_layer(observations_actions)


class DDPGModel(tf.Module):

    def __init__(self, actor, critic, observation_shape):
        self.actor = actor
        self.critic = critic
        self.actor_lr = 5e-4
        self.critic_lr = 5e-4

        self.target_actor = Actor(observation_shape=observation_shape)
        self.target_critic = Critic(observation_shape=observation_shape)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    @tf.function
    def step(self, obs):
        pass

    def train(self):
        pass




def train(env, conf):

    critic = Critic(observation_shape=env.observation_space.shape)
    actor = Actor(observation_shape=env.observation_space.shape)

    ddpg = DDPGModel(actor, critic, env.observation_space.shape)


def val(env, conf):
    # TODO
    pass
