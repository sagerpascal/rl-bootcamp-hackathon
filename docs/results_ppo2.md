---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /results_ppo2
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

# Results Deep PPO (Clip-PPO)

## Network Architecture of the Policy Network
PPO was the second method I implemented. To get started, I used the same network architecture as I used for DQN.
Compared to DQN, smaller networks (less neurons) worked better for PPO: 

<img src="\rl-bootcamp-hackathon\assets\images\ppo2\different_architecture.png" alt="different_architecture"/>

From these network architectures I choose the nw3 with more layers and less neurons since it was more stable than the one with only more layers.
This network is very simple, has only 3'456 parameters and looks as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 8)]               0         
_________________________________________________________________
dense (Dense)                (None, 32)                288       
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_3 (Dense)              (None, 32)                1056      
=================================================================
Total params: 3,456
Trainable params: 3,456
Non-trainable params: 0
_________________________________________________________________
```

## Unstable Loss
The loss of the policy network was highly fluctuating. This was due to the fact I didn't use a (appropriate) distribution function.
Although I didn't understand this part in detail, I copied it from OpenAI Baseline. This implementation takes a policy distribution from the latent space of the policy network, and calculates the negative log probability. The ratio of the change in this value is then incorporated into the loss. This has significantly improved the implementation and stabilized the loss:

<img src="\rl-bootcamp-hackathon\assets\images\ppo2\loss_policy_net.png" alt="loss_policy_net"/>

## Parallel Environment
PPO was much slower than DQN. In the code of OpenAI baseline I noticed that they offer MPI support. With MPI, multiple environments are run im parallel and the gradients are averaged. I adopted this part of the code and compared the performance.
For up to 5 parallel environments, the FPS rate increased continuously. For more environments, the FPS-rate didn't increase anymore because no more CPU resources were available.

<img src="\rl-bootcamp-hackathon\assets\images\ppo2\fps.png" alt="fps"/>

However, this is not enough to win the game faster:

<img src="\rl-bootcamp-hackathon\assets\images\ppo2\fps_reward.png" alt="fps_reward"/>

**However, the higher frame rate results in more gradients at the same time. The optimizer can process them in parallel. This leads to more stable results and allows an increase of the learning rate.**

## Hyperparameter Tuning

Many different parameters could be tried out. However, the most influential ones are:
- Clip-Range
- Gamma
- Learning rate
- Batch Size

I have therefore limited myself to these parameters. First, I performed tests with the individual parameters to make sure that they have the expected influence and also to get a feeling for them.

**Different Clip-Range:**
<img src="\rl-bootcamp-hackathon\assets\images\ppo2\different_clip_range.png" alt="different_clip_range"/>

**Different Learning-Rate:**
<img src="\rl-bootcamp-hackathon\assets\images\ppo2\different_lr.png" alt="different_lr"/>

**Different Gamma:**
<img src="\rl-bootcamp-hackathon\assets\images\ppo2\different_gamma.png" alt="different_gamma"/>

After this exploration I wanted to run sweeps, but didn't have enough time. It is likely that the performance could be optimized even further.
Additionally, other hyperparameters such as the batch size should also be examined.

## Stability
One advantage of PPO compared to DQN is the stability. The reward increases slower, but overall it has much less fluctuations than DQN.

*Run over 4.5h*
<img src="\rl-bootcamp-hackathon\assets\images\ppo2\stability.png" alt="stability"/>


