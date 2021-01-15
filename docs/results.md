---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /results
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

<br>
<br>

### Results LunarLanderContinuous-v2

| **Network**  | **Duration until Won** | **Max. mean Score** |
|--------------|------------------------|---------------------|
| DDPG         | Not won yet!           | 0                   |


<br>

## Results of the different Algorithms:
- [Results DQN](/rl-bootcamp-hackathon/results_dqn)
- [Results PPO](/rl-bootcamp-hackathon/results_ppo2)
- [Results A2C](/rl-bootcamp-hackathon/results_a2c)
- [Results DDPG](/rl-bootcamp-hackathon/results_ddpg)
