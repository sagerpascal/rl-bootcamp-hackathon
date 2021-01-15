---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /results_a2c
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
- [Conclusion](/rl-bootcamp-hackathon/conclusion)
- [Presentation](/rl-bootcamp-hackathon/presentation)



# Results Advantage Actor Critic (A2C)
A2C was the third network I implemented. The main goal was to implement a policy optimization method that learns as fast as DQN. 
Therefore, I put the focus more on speed than stability.

## Network Architecture
I have tried different, relatively small networks. In the end, a network with 2 layers and 64 units each achieved the best performance.

<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_architecture.png" alt="different_architecture"/>

The used network looks as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 8)]               0         
_________________________________________________________________
dense (Dense)                (None, 64)                576       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
=================================================================
Total params: 4,736
Trainable params: 4,736
Non-trainable params: 0
_________________________________________________________________
```

## Parallel Environment
Since I implemented parallel environments for PPO2 (see [Results PPO](/results_ppo2)), I used this for A2C as well:

**FPS:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_envs_fps.png" alt="different_envs_fps"/>

**Reward:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_envs_reward.png" alt="different_envs_reward"/>

## Different Optimizers and Schedulers

I tried different combinations of optimizers and schedulers (only a few of them plotted below):

**Different Optimizers:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_optimizers.png" alt="different_optimizers"/>

**Different Schedulers:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_scheduler.png" alt="different_scheduler"/>

In the end I was a little lucky and found a combination that is bit unstable but very fast. Since I wanted to achieve a 
solution which wins the game as fast as possible I used this combination of linear scheduler and RMSprop optimizer.

## Hyperparameters
I tried different hyperparameters, but did not do any tuning with grid search or random search. Nevertheless, this was enough to win the game rather quickly.

**Different Gamma:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_gamma.png" alt="different_gamma"/>

**Different Learning Rate:**
<img src="\rl-bootcamp-hackathon\assets\images\a2c\different_lr.png" alt="different_lr"/>

# Video of the Result
<video width="600" height="450" controls>
  <source src="\rl-bootcamp-hackathon\assets\videos\a2c\a2c.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>