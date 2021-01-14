---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /results_dqn
---

# Navigation
- [Home](/rl-bootcamp-hackathon/)
- [Procedure](/rl-bootcamp-hackathon/procedure)
- [Algorithms](/rl-bootcamp-hackathon/algorithms)
- [Results](/rl-bootcamp-hackathon/results)
  - [Results DQN](/rl-bootcamp-hackathon/results_dqn)
  - [Results PPO](/rl-bootcamp-hackathon/results_ppo2)
  - [Results A2C](/rl-bootcamp-hackathon/results_a2c)
  - [Results DDPG](/rl-bootcamp-hackathon/results_ddpg)
- [Presentation](/rl-bootcamp-hackathon/presentation)



# Results Deep Q-Net

## Different Network Architectures

### Activation Function
After implementing an initial version, I started testing different network architectures. I did this for two reasons:
- The result was very bad (game could not be won).
- The reward was very unstable (strong fluctuations)

One of the biggest improvements was achieved by using a different activation function. After switching from `tanh` to `relu`, the 
reward was much more stable *(note: this plot was recorded only after an initial parameter tuning, so a reward of >200 was already achieved)*:

<img src="\rl-bootcamp-hackathon\assets\images\deepq\different_activation.png" alt="different_activation"/>

**Overall, using ReLU has led to much more stable results!**

### Different Number of Layers and Neurons
I have also tried different architectures. However, only a few architectures could be tested due to the limited time. 
Starting from the basic version, I tried different number of layers and neurons.
I compared the networks by measuring the reward per time (and not per episode). I did this because I wanted to favor 
higher performing networks. However, the result is for both versions (per time or per episods) almost the same.

<img src="\rl-bootcamp-hackathon\assets\images\deepq\different_architectures.png" alt="different_architectures"/>

The best result was achieved with a rather small network with 22,021 parameters.

## Different Loss Functions
To further improve the result I read trough the article [https://.manning.com//grokking-deep-reinforcement-learning//](https://livebook.manning.com/book/grokking-deep-reinforcement-learning/chapter-6/v-4/)
and also compared my implementation with the OpenAI baseline. Both sources have recommended a Huber loss-function. 
Therefore I compared this loss function with the MSE loss which was used in the original paper ([https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)).

<img src="\rl-bootcamp-hackathon\assets\images\deepq\square_loss_vs_huber_loss.png" alt="square_loss_vs_huber_loss"/>

Unfortunately, this only brought a slight improvement, but since I had already implemented the Huber loss function, I kept it.

## Double Q-Learning
I also compared Q-Learning with Double Q-Learning. 
> Because the future maximum approximated action value in Q-learning is evaluated using the same Q function as in current action selection policy, in noisy environments Q-learning can sometimes overestimate the action values, slowing the learning. A variant called Double Q-learning was proposed to correct this. (source: [https://en.wikipedia.org/wiki/Q-learning](https://en.wikipedia.org/wiki/Q-learning)) 

<img src="\rl-bootcamp-hackathon\assets\images\deepq\Q_Learning_vs_Double_Q_Learning.png" alt="Q_Learning_vs_Double_Q_Learning"/>

**Double Q-learning has mainly made the result more stable.**

## Tuning Hyper-Parameters
Many different parameters could be tried out. However, the most influential ones are:
- Buffer size
- Gamma
- Learning rate
- Epsilon Decay

I have therefore limited myself to these parameters. First, I performed tests with the individual parameters to make sure that they have the expected influence and also to get a feeling for them.

**Different Buffer Size:**
<img src="\rl-bootcamp-hackathon\assets\images\deepq\different_buffer_size.png" alt="different_buffer_size"/>

**Different Gamma:**
<img src="\rl-bootcamp-hackathon\assets\images\deepq\mean_reward_different_gamma.png" alt="mean_reward_different_gamma"/>

**Different Learning Rate:**
<img src="\rl-bootcamp-hackathon\assets\images\deepq\different_lr.png" alt="different_lr"/>

**Epsilon Decay:**
<img src="\rl-bootcamp-hackathon\assets\images\deepq\eps_decay_reward.png" alt="eps_decay_reward"/>
<img src="\rl-bootcamp-hackathon\assets\images\deepq\eps_decay_exploration_time.png" alt="eps_decay_exploration_time"/>

After this exploration I wanted to run sweeps, but didn't have enough time. It is likely that the performance could be optimized even further.
Additionally, other hyperparameters such as the batch size should also be examined.

# Stability
DQN achieves good results relatively quickly, but one problem is stability. After the game is won for the first time, the algorithm has strong fluctuations. So far, it has not been found out what exactly causes this. Some modifications like the adjustment of the loss function have reduced these fluctuations, but they are still strong.

<img src="\rl-bootcamp-hackathon\assets\images\deepq\stability.png" alt="stability"/>