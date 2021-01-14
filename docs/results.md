---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /results
---

# Navigation
- [Home](/)
- [Procedure](/procedure)
- [Algorithms](/algorithms)
- [Presentation](/presentation)

# Results

## Comparison of Different Algorithms
My focus was to win the game as quickly as possible. I measured the time and not the number of episodes. The episodes vary  depending on the algorithm and therefore are not relevant.
For example, DQN takes many episodes even though the game is won in 10 minutes. But this is because I wait a little longer to fill the buffer until I start learning.

### Results LunarLander-v2

*Won = mean return > 200 for 100 epochs min.*

| **Network**  | **Duration until Won** | **Max. mean Score** |
|--------------|------------------------|---------------------|
| DQN          | 10min 41s (GTX 1060)   | 268                 |
| PPO2         | 22min 15s (Tesla T4)   | 245                 |
| A2C          |  4min 40s (GTX 1060)   | 226                 |

### Results LunarLanderContinuous-v2

| **Network**  | **Duration until Won** | **Max. mean Score** |
|--------------|------------------------|---------------------|
| DDPG         | Not won yet!           | 0                   |


## Results of the different Algorithms:
- [Results DQN](/results_dqn)
- [Results PPO](/results_ppo2)
- [Results A2C](/results_a2c)
- [Results DDPG](/results_ddpg)
