---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /procedure
---

# Navigation
- [Home](/rl-bootcamp-hackathon/)
- [Algorithms](/rl-bootcamp-hackathon/algorithms)
- [Results](/rl-bootcamp-hackathon/results)
- [Presentation](/rl-bootcamp-hackathon/presentation)


# Definition of the Problem Space
In this hackathon, we need to implement a RL solution for the OpenAI environment **"LunarLander-v2"**.

Description of LunarLander-v2 according [https://gym.openai.com/envs/LunarLander-v2/](https://gym.openai.com/envs/LunarLander-v2/):
  - *Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.*

    
In addition to the LunarLander-v2 environment, the LunarLanderContinuous-v2 environment can also be solved. This environment has compared to the first version a continuous action space. 

### Environment properties:
- Actions: The left and the right engine from the Lander can be activated (two floats, given by gym environment)
- States: The state of the environment is encoded as vector:
    - Index 0: x-position of the lander
    - Index 1: y-position of the lander
    - Index 2: x velocity
    - Index 3: y-velocity
    - Index 4: Lander angle
    - Index 5: Lander angle velocity
    - Index 6: Left leg has ground contact
    - Index 7: Right leg has ground contact



# Selection of the algorithm

*Source: [https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)*

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

I selected the algorithms based on the environment. Since the reward function is well defined, I have **not considered
 inverse reinforcement learning**. Next, I evaluated whether model-based or model-free reinforcement learning should be used.
[https://spinningup.openai.com/](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#model-free-vs-model-based-rl)
describes this branching point as follows:
> One of the most important branching points in an RL algorithm is the question of whether the agent has access to (or learns) a model of the environment. By a model of the environment, we mean a function which predicts state transitions and rewards.

For the LunarLander environment, model-free reinforcment learning is more appropriate. The reason is that the model cannot be described or learned in a simple way. Therefore I limited my implementation to the left side of the algorithms in the image above.

### Different Approaches

There are two main approaches used in model-free RL:
- **Policy Optimization.** Methods in this family represent a policy explicitly as ![\pi_{\theta}(as)](https://spinningup.openai.com/en/latest/_images/math/400068784a9d13ffe96c61f29b4ab26ad5557376.svg). They optimize the parameters ![\theta](https://spinningup.openai.com/en/latest/_images/math/ce5edddd490112350f4bd555d9390e0e845f754a.svg) either directly by gradient ascent on the performance objective ![J(\pi_{\theta})](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg), or indirectly, by maximizing local approximations of ![J(\pi_{\theta})](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg). This optimization is almost always performed **on-policy**, which means that each update only uses data collected while acting according to the most recent version of the policy. Policy optimization also usually involves learning an approximator ![V_{\phi}(s)](https://spinningup.openai.com/en/latest/_images/math/693bb706835fbd5903ad9758837acecd07ef13b1.svg) for the on-policy value function ![V^{\pi}(s)](https://spinningup.openai.com/en/latest/_images/math/a81303323c25fc13cd0652ca46d7596276e5cb7e.svg), which gets used in figuring out how to update the policy.
- **Q-Learning.** Methods in this family learn an approximator ![Q_{\theta}(s,a)](https://spinningup.openai.com/en/latest/_images/math/de947d14fdcfaa155ef3301fc39efcf9e6c9449c.svg) for the optimal action-value function, ![Q^*(s,a)](https://spinningup.openai.com/en/latest/_images/math/cbed396f671d6fb54f6df5c044b82ab3f052d63e.svg). Typically they use an objective function based on the [Bellman equation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations). This optimization is almost always performed **off-policy**, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained.

The primary strength of policy optimization methods is that they are principled, in the sense that you directly optimize for the thing you want. This tends to make them stable and reliable. By contrast, Q-learning methods only indirectly optimize for agent performance, by training <img src="https://render.githubusercontent.com/render/math?math=Q_{\theta}"> to satisfy a self-consistency equation. There are many failure modes for this kind of learning, so it tends to be less stable. But, Q-learning methods gain the advantage of being substantially more sample efficient when they do work, because they can reuse data more effectively than policy optimization techniques.
Serendipitously, policy optimization and Q-learning are not incompatible (and under some circumstances, it turns out, equivalent), and there exist a range of algorithms that live in between the two extremes. Algorithms that live on this spectrum are able to carefully trade-off between the strengths and weaknesses of either side.

## Used Algorithm
In the end, both methods (Policy Optimization and Q-Learning) have their advantages.  Therefore, I did not know which approach is more suitable. A mixture of the two approaches could also be promising. 
During the hackathon, I decided to use the following algorithms:

- **Deep Q-Learning**: I started with Deep Q-Learning as baseline. I chose Deep Q-Learning because we have already discussed this algorithm in class and had a good feeling about it.
  - Besides DQN, there are some exciting alternatives of Q-Learning algorithms, for example **C51**. The main difference between C51 and DQN is that C51 not only predicts the Q-value for each state-action pair, but predicts a histogram model for the probability distribution of the Q-value (source [Tensorflow C51 Tutorial](https://www.tensorflow.org/agents/tutorials/9_c51_tutorial)). The same applies to **QR-DQN**, which can be regarded as a further improvement of C51. (source [Distributional Reinforcement Learning](https://medium.com/analytics-vidhya/distributional-reinforcement-learning-part-1-c51-and-qr-dqn-a04c96a258dc))
  - Another alternative is **Hindsight Experience Replay (HER)**. This algorithm is particularly well suited for sparse rewards. HER extends DQN with a kind of "data augmentation". The unique idea behind HER is to replay each episode with a different goal than the one the agent was initially trying to achieve. (source [Advanced Exploration: Hindsight Experience Replay](https://medium.com/analytics-vidhya/advanced-exploration-hindsight-experience-replay-fd604be0fc4a))
  - *Although Ithere are newer and better performing algorithms than DQN, I have decided to use DQN. This is mainly because I already know and understand DQN. However, if there is enough time in the end, it would be very interesting to either extend DQN with the idea of HER or to implement QR-DQN. But I don't want to invest too much time in this, since statistics show that for this specific environment not much better results can be expected than with Q-Learning.*
- **Proximal Policy Optimization**: As a second algorithm, I wanted to implement a policy optimization method. In class, we mainly covered TRPO. "PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO." (source: [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)). That's why I decided to implement PPO instead of TRPO.
- **Advantage Actor Critic**: The results from PPO were good but it took much longer to win the game with PPO than with DQN.  I didn't know if this was due to the implementation, the algorithm or if the Policy Optimization methods in general worked worse for this problem. Therefore, I chose A2C as the third algorithm. I also wanted to use Actor-Critic because we discussed it in class and it is a hybrid of Policy Optimization and Dynamic Programming. There are different implementations of Actor-Critic. I used the version "Advantage Actor-Critic", which is a synchronous (single worker) implementation of A3C. According to my research, this version is widely used and achieves good results for the given environment.
- **Deep Deterministic Policy Gradient**: Since I haven't yet implemented a mixture of policy optimization and Q-learning, I wanted to finish with this. I also had not yet implemented an algorithm for the continuous action space (LunarLandingContinous-V2). After a short research I found a good solution to cover both was the use of DDPG. I also chose DDPG because this algorithm contains many concepts I already know like actor-critic or ring buffers.

# Implementation
First, I set up Docker on the GPU-cluster so I could debug locally and run experiments on the cluster. After that, my approach was identical for most algorithms:

1. implementation of a simple base version
2. debug & check results
3. comparison with high-end implementation (Stable Baseline, OpenAI Baseline, Dopamine, Horizon, ...)
4. adopt advanced concepts (policy distribution, latent space, ...)
5. record tests and results
