import os
import random

import numpy as np
import tensorflow as tf

from utils import loghandler

"""
Implementation of Deep Q-Learning
    
    Deep Q Network (DQN) builds on Fitted Q-Iteration (FQI) and make use of different tricks to stabilize the learning 
    with neural networks: it uses a replay buffer, a target network and gradient clipping.
    
----------------------------------------------

Created:
    13.01.2021, Pascal Sager <sage@zhaw.ch>

Paper:
    Playing Atari with Deep Reinforcement Learning (https://arxiv.org/abs/1312.5602)

Introduction to Q-Learning:
    https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/

Code-Sources:
    https://github.com/keon/deep-q-learning
    https://github.com/hill-a/stable-baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    https://github.com/openai/baselines

"""


@tf.function
def huber_loss_function(error, delta=1.0):
    """
    Huber loss-function to increase stability of the DQN algorithm
    (see https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-6/v-4/)

    Reference: https://en.wikipedia.org/wiki/Huber_loss
    """
    return tf.where(
        tf.abs(error) < delta,
        tf.square(error) * 0.5,
        delta * (tf.abs(error) - 0.5 * delta)
    )


class EpsScheduler:
    """
    Scheduler to reduce the exploration rate linear from n initial_eps to final_eps over
    number_of_timesteps
    """

    def __init__(self, number_of_timesteps, initial_eps, final_eps):
        self.schedule_timesteps = number_of_timesteps
        self.final_p = final_eps
        self.initial_p = initial_eps

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def build_q_function_network():
    """
    Returns the network used for the prediction and target network
    """

    def q_func_builder(input_shape, num_actions):
        # base network without top layer
        input = tf.keras.Input(input_shape)
        base_layers = input
        for i in range(2):
            base_layers = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(base_layers)

        input_network = tf.keras.Model(inputs=[input], outputs=[base_layers])

        # extend base network with two heads: action-value network and state-value network
        latent = tf.keras.layers.Flatten()(input_network.outputs[0])

        with tf.name_scope("state-value"):
            state_head = latent
            state_head = tf.keras.layers.Dense(units=256, activation=None)(state_head)
            state_head = tf.nn.relu(state_head)
            state_score = tf.keras.layers.Dense(units=1, activation=None)(state_head)

        with tf.name_scope("action-value"):
            action_head = latent
            action_head = tf.keras.layers.Dense(units=256, activation=None)(action_head)
            action_head = tf.nn.relu(action_head)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_head)

        action_scores_mean = tf.reduce_mean(action_scores, 1)
        action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
        q_out = action_scores_centered + state_score

        model = tf.keras.Model(inputs=input_network.inputs, outputs=[q_out])
        print(model.summary())
        return model

    return q_func_builder


class DeepQLearning(tf.Module):

    def __init__(self, q_function, observations_shape, number_of_actions, lr, gamma):

        self.number_of_actions = number_of_actions
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(lr)

        # Since the same network is calculating the predicted value and the target value, there could be a lot of
        # divergence between these two. So, instead of using 1one neural network for learning, we can use two.
        #
        # We could use a separate network to estimate the target. This target network has the same architecture as the
        # function approximator but with frozen parameters. For every C iterations (a hyperparameter), the parameters
        # from the prediction network are copied to the target network. This leads to more stable training because
        # it keeps the target function fixed (for a while)
        with tf.name_scope('q_network'):
            self.q_network = q_function(observations_shape, number_of_actions)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_function(observations_shape, number_of_actions)
        self.eps = tf.Variable(0., name="eps")

    @tf.function
    def step(self, obs, eps=-1):
        """
        Select an action:
        First, get the Q-values from the Q-Network. Then calculate the best possible action (argmax).
        Then either select this best possible action or a random action (depending on eps, eps reduces over time).
        """
        q_values = self.q_network(obs)
        best_actions = tf.argmax(q_values, axis=1)
        batch_size = tf.shape(obs)[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.number_of_actions,
                                           dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        actions = tf.where(chose_random, random_actions, best_actions)
        if eps >= 0:
            self.eps.assign(eps)

        return actions

    @tf.function()
    def train(self, observation, actions, rewards, new_observation, dones, importance_weights):
        """
        Update the network parameters:
        First calculate the Q-Values from the Prediction network. Then calculate the Q-Values from the target network
        and compare it with them from the prediction network. The difference between them is the temporal difference (td).
        Then calculate the error (apply loss function on td) and update the gradients accordingly.
        """

        with tf.GradientTape() as tape:
            q_values = self.q_network(observation)
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, self.number_of_actions, dtype=tf.float32),
                                              1)

            q_values_target = self.target_q_network(new_observation)

            # Double Q-Learning: Use online and offline network
            q_values_target_online = self.q_network(new_observation)
            best_q_values_target_online = tf.argmax(q_values_target_online, 1)
            best_q_values_online = tf.reduce_sum(
                q_values_target * tf.one_hot(best_q_values_target_online, self.number_of_actions, dtype=tf.float32), 1)

            # Without Double Q-Learning
            # best_q_values_online = tf.reduce_max(q_values_target, 1)

            dones = tf.cast(dones, best_q_values_online.dtype)
            best_q_values_online_without_dones = (1.0 - dones) * best_q_values_online

            q_values_target_selected = rewards + self.gamma * best_q_values_online_without_dones

            # Calculate the temporal difference error and weight this error
            td_error = q_values_selected - tf.stop_gradient(q_values_target_selected)
            errors = huber_loss_function(td_error)
            weighted_error = tf.reduce_mean(importance_weights * errors)

        gradients = tape.gradient(weighted_error, self.q_network.trainable_variables)
        gradients_variables = zip(gradients, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(gradients_variables)

        return td_error

    @tf.function(autograph=False)
    def update_target(self):
        """
        Assign the trainable variables from the prediction network to the target network
        """
        q_network_variables = self.q_network.trainable_variables
        target_q_network_variables = self.target_q_network.trainable_variables
        for variables_qn, variables_qn_target in zip(q_network_variables, target_q_network_variables):
            variables_qn_target.assign(variables_qn)


class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer_size = size
        self.index = 0
        self.buffer_items = []

    def __len__(self):
        return len(self.buffer_items)

    def add(self, observation, action, reward, new_observation, done):
        data = (observation, action, reward, new_observation, done)

        if self.index >= len(self.buffer_items):
            self.buffer_items.append(data)
        else:
            self.buffer_items[self.index] = data

        self.index = (self.index + 1) % self.buffer_size

    def get_minibatch(self, batch_size):
        """
        Sample a batch of experiences.
        """
        index_list = [random.randint(0, len(self.buffer_items) - 1) for _ in range(batch_size)]
        observations, actions, rewards, new_observations, dones = [], [], [], [], []
        data = self.buffer_items[0]

        for i in index_list:
            data = self.buffer_items[i]
            observation, action, reward, new_observation, done = data
            observations.append(np.array(observation, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            new_observations.append(np.array(new_observation, copy=False))
            dones.append(done)
        return np.array(observations, dtype=data[0].dtype), np.array(actions, dtype=data[1].dtype), \
               np.array(rewards, dtype=np.float32), \
               np.array(new_observations, dtype=data[0].dtype), np.array(dones, dtype=np.float32)


def load_checkpoints(conf, model):
    if conf['load_path'] is not None:
        load_path = os.path.expanduser(conf['load_path'])
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))


def train(env, conf):
    logger = loghandler.LogHandler(conf, trigger='episodes')
    q_func = build_q_function_network()

    model = DeepQLearning(
        q_function=q_func,
        observations_shape=env.observation_space.shape,
        number_of_actions=env.action_space.n,
        lr=conf['lr'],
        gamma=conf['gamma'],
    )

    load_checkpoints(conf, model)
    replay_buffer = ReplayBuffer(conf['buffer_size'])
    exploration_scheduler = EpsScheduler(
        number_of_timesteps=int(conf['exploration_fraction'] * conf['total_timesteps']),
        initial_eps=1.0, final_eps=conf['exploration_final_eps'])

    model.update_target()
    episode_rewards = [0.0]
    observation = env.reset()
    best_reward = -100000000000

    for t in range(conf['total_timesteps']):
        kwargs = {}
        logs = {}
        updated_eps_value = tf.constant(exploration_scheduler.value(t))

        action = model.step(tf.constant(observation), eps=updated_eps_value, **kwargs)
        action = action[0].numpy()
        new_observation, reward, done, _ = env.step(action)

        replay_buffer.add(observation[0], action, reward[0], new_observation[0], float(done[0]))
        observation = new_observation

        episode_rewards[-1] += reward
        if done:
            observation = env.reset()
            episode_rewards.append(0.0)

        # Update the network (minimize error in Bellman's equation)
        if t > conf['learning_starts'] and t % conf['train_freq'] == 0:
            observation, actions, rewards, new_observation, dones = replay_buffer.get_minibatch(conf['batch_size'])
            weights = np.ones_like(rewards)
            observation, new_observation = tf.constant(observation), tf.constant(new_observation)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            weights = tf.constant(weights)
            td_errors = model.train(observation, actions, rewards, new_observation, dones, weights)

        # Copy parameters to target network
        if t > conf['learning_starts'] and t % conf['target_network_update_freq'] == 0:
            model.update_target()

        mean_reward_100_episods = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)

        logs["steps"] = t
        logs["episodes"] = num_episodes
        logs["mean reward in 100 episodes"] = mean_reward_100_episods
        logs["% time spent exploring"] = int(100 * exploration_scheduler.value(t))
        logger.log(logs)

        if t > 150000 and mean_reward_100_episods > best_reward:
            best_reward = mean_reward_100_episods
            save_path = os.path.expanduser('best_models/deepq')
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
            manager.save()


def val(env, conf):
    obs = env.reset()

    if conf['load_path'] is None or conf['load_path'] is 'None':
        raise AttributeError("Load path must be defined to validate model!!!")

    q_func = build_q_function_network()
    model = DeepQLearning(
        q_function=q_func,
        observations_shape=env.observation_space.shape,
        number_of_actions=env.action_space.n,
        lr=conf['lr'],
        gamma=conf['gamma'],
    )

    load_checkpoints(conf, model)
    episode_rew = np.zeros(env.num_envs)
    while True:
        actions = model.step(obs)
        obs, rew, done, _ = env.step(actions.numpy())
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0
