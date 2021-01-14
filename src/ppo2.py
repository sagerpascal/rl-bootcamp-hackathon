import os
import time
from collections import deque

import numpy as np
import tensorflow as tf

from utils.distributions import CategoricalPdType
from utils.loghandler import LogHandler

"""
Implementation Proximal Policy Optimization (PPO-Clip):

    PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy 
    using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where 
    TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that 
    use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, 
    and empirically seem to perform at least as well as TRPO.
    
    PPO-Clip doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on 
    specialized clipping in the objective function to remove incentives for the new policy to get far from the old 
    policy.

----------------------------------------------

Created:
    13.01.2021, Pascal Sager <sage@zhaw.ch>

Paper:
    Proximal Policy Optimization Algorithms (https://arxiv.org/abs/1707.06347)

Code-Sources:
    https://github.com/hill-a/stable-baselines
    https://github.com/openai/baselines

"""

# Try to use Multi Processor Interface
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def explained_variance(y_pred, y):
    """
    Returns 1 - Var[y - y_pred] / Var[y]
    """
    assert y.ndim == 1 and y_pred.ndim == 1
    variance = np.var(y)
    return np.nan if variance == 0 else 1 - np.var(y - y_pred) / variance


class PolicyValueFunctionNetwork(tf.Module):
    """
    The neural network which estimates the policy and the value function
    """
    def __init__(self, action_space, ob_space):

        in_layer = tf.keras.Input(shape=ob_space.shape)
        layers = in_layer
        for i in range(4):
            layers = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(layers)

        self.policy_network = tf.keras.Model(inputs=[in_layer], outputs=[layers])
        print(self.policy_network.summary())
        self.policy_distribution = CategoricalPdType(self.policy_network.output_shape, action_space.n, init_scale=0.01)

        with tf.name_scope('value function'):
            layer = tf.keras.layers.Dense(units=1, bias_initializer=tf.keras.initializers.Constant(0.0))
            layer.build(self.policy_network.output_shape)
        self.value_fc = layer

    @tf.function
    def step(self, observation):
        """
        Calculate next action given the observation
        """
        latent = self.policy_network(observation)
        policy_distribution, _ = self.policy_distribution.pdfromlatent(latent)
        action = policy_distribution.sample()
        negative_log_prob = policy_distribution.neglogp(action)
        vf = tf.squeeze(self.value_fc(latent), axis=1)
        return action, vf, negative_log_prob

    @tf.function
    def value(self, observation):
        """
        Calculate value estimate given the observation
        """
        latent = self.policy_network(observation)
        result = tf.squeeze(self.value_fc(latent), axis=1)
        return result


def swap_and_flatten(array):
    """
    swap and then flatten axes 0 and 1
    """
    arr_shape = array.shape
    return array.swapaxes(0, 1).reshape(arr_shape[0] * arr_shape[1], *arr_shape[2:])


class PPO2Model(tf.Module):
    def __init__(self, ac_space, ob_space):
        super().__init__(name='ppo2_model')
        self.policy_value_fun_model = PolicyValueFunctionNetwork(ac_space, ob_space)
        self.step = self.policy_value_fun_model.step
        self.value = self.policy_value_fun_model.value

        if MPI is not None:
            from utils.optimizers import MpiAdamOptimizer
            from utils.multiproccessor_utils import sync_from_root
            self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.policy_value_fun_model.trainable_variables)
            sync_from_root(self.variables)
        else:
            self.optimizer = tf.keras.optimizers.Adam()


    def train(self, lr, cliprange, obs, returns, actions, values, negative_log_probability_actions_old):
        """
        Train the network: Calculate the gradients and the apply the optimizer
        """
        grads, pg_loss, vf_loss = self.get_grad(cliprange, obs, returns, actions, values, negative_log_probability_actions_old)
        if MPI is not None:
            self.optimizer.apply_gradients(grads, lr)
        else:
            self.optimizer.learning_rate = lr
            grads_and_vars = zip(grads, self.policy_value_fun_model.trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss

    @tf.function
    def get_grad(self, cliprange, obs, returns, actions, values, negative_log_probability_actions_old):
        """
        Calculation of the gradients

        """
        # calculate advantage A(s,a) = R + yV(s') - V(s) and then normalize it
        advantage = returns - values
        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.keras.backend.std(advantage) + 1e-8)

        with tf.GradientTape() as tape:
            # Calculate the policy distribution
            policy_latent = self.policy_value_fun_model.policy_network(obs)
            policy_distribution, _ = self.policy_value_fun_model.policy_distribution.pdfromlatent(policy_latent)
            negative_log_probability_actions = policy_distribution.neglogp(actions)
            # Predict the values
            predicted_values = self.policy_value_fun_model.value(obs)
            # Clip the values
            clipped_values = values + tf.clip_by_value(predicted_values - values, -cliprange, cliprange)

            # Calculate the loss of the value function
            value_func_loss_1 = tf.square(predicted_values - returns)
            value_func_loss_2 = tf.square(clipped_values - returns)
            loss_value_function = 0.5 * tf.reduce_mean(tf.maximum(value_func_loss_1, value_func_loss_2))

            # Calculate the loss of the policy
            ratio = tf.exp(negative_log_probability_actions_old - negative_log_probability_actions)
            policy_loss_1 = -advantage * ratio
            policy_loss_2 = -advantage * tf.clip_by_value(ratio, 1 - cliprange, 1 + cliprange)
            policy_loss = tf.reduce_mean(tf.maximum(policy_loss_1, policy_loss_2))

            loss = policy_loss + loss_value_function * 0.5

        variables = self.policy_value_fun_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        max_grad_norm = 0.5
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        if MPI is not None:
            gradients = tf.concat([tf.reshape(g, (-1,)) for g in gradients], axis=0)
        return gradients, policy_loss, loss_value_function


def load_checkpoints(conf, model):
    if conf['load_path'] is not None:
        load_path = os.path.expanduser(conf['load_path'])
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)


def mean_or_nan(val):
    """
    return nan if value is empty else the mean
    """
    return np.nan if len(val) == 0 else np.mean(val)


class MinibatchExecuter:
    """
    Create a minibatch of samples
    """

    def __init__(self, environment, model, number_of_steps, gamma, lambda_):
        self.environment = environment
        self.model = model
        self.number_of_envs = environment.num_envs if hasattr(environment, 'num_envs') else 1
        self.batch_ob_shape = (self.number_of_envs * number_of_steps,) + environment.observation_space.shape
        self.observations = np.zeros((self.number_of_envs,) + environment.observation_space.shape,
                                     dtype=environment.observation_space.dtype.name)
        self.observations[:] = environment.reset()
        self.number_of_steps = number_of_steps
        self.dones = [False for _ in range(self.number_of_envs)]
        # Lambda used in GAE (General Advantage Estimation)
        self.lambda_ = lambda_
        # Discount rate
        self.gamma = gamma

    def run(self):
        observations_minibatch, rewards_minibatch, actions_minibatch, values_minibatch, dones_minibatch, neg_log_probs_actions_minibatch = [], [], [], [], [], []
        episode_infos = []

        for _ in range(self.number_of_steps):
            # execute a step for a given observation (initialized by env.reset)
            actions, values, negative_log_prob_actions = self.model.step(tf.constant(self.observations))
            observations_minibatch.append(self.observations.copy())
            actions = actions._numpy()
            actions_minibatch.append(actions)
            values_minibatch.append(values._numpy())
            neg_log_probs_actions_minibatch.append(negative_log_prob_actions._numpy())
            dones_minibatch.append(self.dones)

            # execute the actions in the environment and read the logs (for wandb)
            self.observations[:], rewards, self.dones, logs = self.environment.step(actions)
            for log in logs:
                episode_log = log.get('episode')
                if episode_log:
                    episode_infos.append(episode_log)
            rewards_minibatch.append(rewards)

        # batch of steps to batch of rollouts
        observations_minibatch = np.asarray(observations_minibatch, dtype=self.observations.dtype)
        rewards_minibatch = np.asarray(rewards_minibatch, dtype=np.float32)
        actions_minibatch = np.asarray(actions_minibatch)
        values_minibatch = np.asarray(values_minibatch, dtype=np.float32)
        neg_log_probs_actions_minibatch = np.asarray(neg_log_probs_actions_minibatch, dtype=np.float32)
        dones_minibatch = np.asarray(dones_minibatch, dtype=np.bool)
        last_values = self.model.value(tf.constant(self.observations))._numpy()

        # discount off value fn
        advantage_minibatch = np.zeros_like(rewards_minibatch)
        lastgaelam = 0
        for t in reversed(range(self.number_of_steps)):
            if t == self.number_of_steps - 1:
                next_non_terminal = 1. - self.dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones_minibatch[t + 1]
                next_values = values_minibatch[t + 1]
            delta = rewards_minibatch[t] + self.gamma * next_values * next_non_terminal - values_minibatch[t]
            advantage_minibatch[t] = lastgaelam = delta + self.gamma * self.lambda_ * next_non_terminal * lastgaelam
        returns_minibatch = advantage_minibatch + values_minibatch
        return (*map(swap_and_flatten, (observations_minibatch, returns_minibatch, actions_minibatch, values_minibatch, neg_log_probs_actions_minibatch)), episode_infos)


def get_model(env):
    ob_space = env.observation_space
    ac_space = env.action_space

    model = PPO2Model(ac_space=ac_space, ob_space=ob_space)
    return model


def train(env, conf):
    logger = LogHandler(conf)

    number_of_envs = env.num_envs
    # Calculate the batch_size
    nbatch = number_of_envs * conf['n_steps_ppo2']
    nbatch_train = nbatch // conf['nminibatches_ppo2']

    model = get_model(env)
    load_checkpoints(conf, model)

    minibatch_executer = MinibatchExecuter(environment=env, model=model, number_of_steps=conf['n_steps_ppo2'], gamma=conf['gamma'],
                               lambda_=0.95)
    info_buffer = deque(maxlen=100) # stores the infos from the episodes

    best_reward = -100000000

    nupdates = conf['total_timesteps_ppo2'] // nbatch
    for update in range(1, nupdates + 1):
        # Start timer
        start_time = time.perf_counter()

        # Get minibatch
        observations, returns, actions, values, negative_log_probability_actions, episode_infos = minibatch_executer.run()  # pylint: disable=E0632
        info_buffer.extend(episode_infos)

        # Calc loss for each minibatch
        loss_values_minibatch = []
        batch_indexes = np.arange(nbatch)
        for _ in range(conf['noptepochs']):
            np.random.shuffle(batch_indexes)  # shuffle -> improves performance!!!
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                i = batch_indexes[start:end]
                slices = (tf.constant(arr[i]) for arr in (observations, returns, actions, values, negative_log_probability_actions))
                loss_values_minibatch.append(model.train(conf['lr'], conf['cliprange'], *slices))


        loss_values = np.mean(loss_values_minibatch, axis=0)
        current_time = time.perf_counter()
        fps = int(nbatch / (current_time - start_time))
        logs = {}
        # Calculates if value function is a good predicator of the returns (ev > 1)
        # or if it's just worse than predicting nothing (ev =< 0)
        ev = explained_variance(values, returns)
        logs["number of updates"] = update
        logs["timestep"] = update * nbatch
        logs["fps"] = fps
        logs["explained variance"] = float(ev)
        mean_reward = mean_or_nan([epinfo['r'] for epinfo in info_buffer])
        logs['mean reward'] = mean_reward
        logs['mean epsiod length'] = mean_or_nan([epinfo['l'] for epinfo in info_buffer])
        for (lossval, lossname) in zip(loss_values,
                                       ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']):
            logs['loss of ' + lossname] = lossval

        logger.log(logs)

        if update > 200 and mean_reward > best_reward:
            best_reward = mean_reward
            save_path = os.path.expanduser('best_models/ppo2')
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
            manager.save()


def val(env, conf):
    obs = env.reset()

    if conf['load_path'] is None or conf['load_path'] is 'None':
        raise AttributeError("Load path must be defined to validate model!!!")

    model = get_model(env)
    load_checkpoints(conf, model)

    episode_reward = np.zeros(env.num_envs)
    while True:
        actions, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(actions.numpy())
        episode_reward += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode reward={}'.format(episode_reward[i]))
                episode_reward[i] = 0
