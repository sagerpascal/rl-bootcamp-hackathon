---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /algorithms
---

# Navigation
- [Home](/rl-bootcamp-hackathon)
- [Procedure](/rl-bootcamp-hackathon/procedure)
- [Results](/rl-bootcamp-hackathon/results)
- [Presentation](/rl-bootcamp-hackathon/presentation)



# Selection of the Algorithms
On this page only the used algorithms are roughly described. How these algorithms were selected is described under [Procedure](/rl-bootcamp-hackathon/procedure).

# On-Policy vs. Off-Policy

PPO is an on-policy algorithm: that is, they don’t use old data, which makes them weaker on sample efficiency. But this is for a good reason: these algorithms directly optimize the objective you care about—policy performance—and it works out mathematically that you need on-policy data to calculate the updates. So, this family of algorithms trades off sample efficiency in favor of stability—but you can see the progression of techniques (from VPG to TRPO to PPO) working to make up the deficit on sample efficiency.



Algorithms like DDPG and Q-Learning are *off-policy*, so they are able to reuse old data very efficiently. They gain this benefit by exploiting Bellman’s equations for optimality, which a Q-function can be trained to satisfy using *any* environment interaction data (as long as there’s enough experience from the high-reward areas in the environment).

But problematically, there are no guarantees that doing a good job of satisfying Bellman’s equations leads to having great policy performance. *Empirically* one can get great performance—and when it happens, the sample efficiency is wonderful—but the absence of guarantees makes algorithms in this class potentially brittle and unstable.





# Deep Q-Learning

In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. The loss function is the mean squared error of the predicted Q-value and the target Q-value. This is basically a regression problem. However, we do not know the target or actual value here as we are dealing with a reinforcement learning problem. But according to the Bellman equation, the Q-value update is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(S_{t+1},A_{t+1}) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_\alpha Q(S_{t+1},a)-Q(S_t,A_t)]">

Since R is the unbiased true reward, the network is going to update its gradient using backpropagation to finally converge. However, there is a challenge when we compare deep RL to deep learning (DL): the target is continuously changing with each iteration. In deep learning, the target variable does not change and hence the training is stable, which is just not true for RL. Since the same network is calculating the predicted value and the target value, there could be a lot of divergence between these two. So, instead of using one neural network for learning, we can use two. We could use a separate network to estimate the target. This target network has the same architecture as the function approximator but with frozen parameters. For every C iterations (a hyperparameter), the parameters from the prediction network are copied to the target network. This leads to more stable training because it keeps the target function fixed (for a while):

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-17-at-12.48.05-PM-768x638.png" style="zoom:50%; margin-left: auto; margin-right: auto; display: block;" />



 Instead of running Q-learning on state/action pairs as they occur during simulation or the actual experience, the system stores the data discovered for [state, action, reward, next_state] – in a large table. This is called **Experience Replay**. 

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-17-at-1.15.28-PM-768x369.png" style="zoom:70%; margin-left: auto; margin-right: auto; display: block;" />

If we put all toegether, we end up with the following procedure:

1. Preprocess and feed the game screen (state <img style="height: 10px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;s">) to the DQN, which will return the Q-values of all possible actions in the state
2. Select an action using the epsilon-greedy policy. With the probability <img style="height: 10px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon">, we select a random action <img style="height: 10px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;a"> and with probability <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;1-\epsilon">, we select an action that has a maximum Q-value, such as <img style="height: 15px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;a = \mathop{\mathrm{argmax}}(Q(s,a,\theta))">
3. Perform this action in a state <img style="height: 10px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;s"> and move to a new state <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;s'"> to receive a reward. This state <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;s'"> is the preprocessed image of the next game screen. We store this transition in our replay buffer as <img style="height: 15px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;<s,a,r,s'>">
4. Next, sample some random batches of transitions from the replay buffer and calculate the loss
5. The loss is defined as
   
   <img style="height: 30px;" src="https://latex.codecogs.com/svg.latex?\Large&space;L=(r+ \gamma \max _{a'}Q(s',a',\theta')-Q(s,a,\theta))^2">
   
   which is just the squared difference between target <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;Q"> and predicted <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;Q">
6. Perform gradient descent with respect to our actual network parameters in order to minimize this loss
7. After every <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;C"> iterations, copy our actual network weights to the target network weights
8. Repeat these steps for <img style="height: 12px; vertical-algin:middle" src="https://latex.codecogs.com/svg.latex?\Large&space;M"> number of episodes



# Proximal Policy Optimization (PPO-clip)

PPO is motivated by the question: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

#### Quick Facts

- PPO is an on-policy algorithm.
- PPO can be used for environments with either discrete or continuous action spaces.


PPO with clipped objective maintains two policy networks. The first one <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_\theta(a|s)"> is the current policy that we want to refine. The second <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_{\theta_k}(a|s)"> is the policy that we last used to collect samples. With the idea of importance sampling, we can evaluate a new policy with samples collected from an older policy. This improves sample efficiency. But as we refine the current policy, the difference between the current and the old policy is getting larger. The variance of the estimation will increase and we will make bad decision because of the inaccuracy. So, after a specific interval (e.g. 4 iterations) we synchronize the second network with the refined policy again. With clipped objective, we compute a ratio between the new policy and the old policy. This ratio measures how difference between two policies. We construct a new objective function to clip the estimated advantage function if the new policy is far away from the old policy. If the probability ratio between the new policy and the old policy falls outside the range <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;(1 - \epsilon)"> and <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;(1+\epsilon)">, the advantage function will be clipped.

*Note: The advantage function is the expected rewards minus a baseline. We use the advantage function instead of the expected reward because it reduces the variance of the estimation. As long as the baseline does not dependent on our policy parameters, the optimal policy will be the same.*



#### Loss Function

PPO-clip updates policies via

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;\theta_{k+1}=\mathop{\mathrm{argmax}}_{\theta} \mathop{\mathrm{E}}_{s,a\sim \pi_{\theta_k}} [L(s,a,\theta_k,\theta)]">

typically taking multiple steps of (usually minibatch) SGD to maximize the objective. Here <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;L"> is the expected advantage function and is given by

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;L(s,a,\theta_k,\theta)=\min \bigg( \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), (g(\epsilon, A^{\pi_{\theta_k}}(s,a))) \bigg)">

where

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;g(\epsilon, A)= \left\{\begin{array}{lr}(1+\epsilon)A, & \text{if}\ A \geq 0 \\ (1-\epsilon) A, & \text{if}\ A < 0\end{array} \right\}">.

<img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon"> is a hyperparameter which roughly says how far away the new policy is allowed to go from the old. <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;A"> ist the advantage, if the advantage increases, the action <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;a"> becomes more likely and if the advantage decreases the action <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;a"> becomes less likely.



What we have seen so far is that clipping serves as a regularizer by removing incentives for the policy to change dramatically, and the hyperparameter <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon"> corresponds to how far away the new policy can go from the old while still profiting the objective. While this kind of clipping goes a long way towards ensuring reasonable policy updates, it is still possible to end up with a new policy which is too far from the old policy, and there are a bunch of tricks used by different PPO implementations to stave this off. In our implementation here, we use a particularly simple method: early stopping. 



#### Exploration vs. Exploitation

PPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.


#### Pseudocode

![https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)



# Advantage Actor-Critic

According to the policy based gradient (see [https://towardsdatascience.com/policy-gradient-step-by-step-ac34b629fd55](https://towardsdatascience.com/policy-gradient-step-by-step-ac34b629fd55)) we have defined an objective function and computed its gradient as follows:

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_\theta J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^T \nabla_{\theta} \log \pi_\theta (a_t|s_t)R_t">



What this equation tells us is that the gradient of <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;J(\theta)"> is the average of all m trajectories, where each trajectory is the sum of the steps that compose it. At each of this step we compute the derivative of the log of the policy <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi"> and multiply it by the return <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;R_t"> In other words we are trying to find how the policy varies following <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\theta">. 

The <img style="height: 15px;" src="https://latex.codecogs.com/svg.latex?\Large&space;R_t"> (return starting at step t) is not bad, but we are not really sure what value of Rt is considered good enough to be taken into consideration?! One way to give a meaning to this number is by comparing it to a reference, or what we call a **baseline**. Baselines can take several forms, one of them is the expected performance or in other terms the average performance. Let’s denote the baseline as <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;b_t">, the gradient of the objective function becomes:

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_\theta J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^T \nabla_{\theta} \log \pi_\theta (a_t|s_t)(R_t-b_t)">

The equation can be rewritten as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_\theta J(\theta)=\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^T \nabla_{\theta} \log \pi_\theta (a_t|s_t)(Q(s_t,a_t)-V_\phi(s_t))">

If we look closely at the equation above, we see that <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_\theta (a_t|s_t)"> is what performs the action (remember <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi"> is the probability of action <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;a"> is taken at state <img style="height: 10px;" src="https://latex.codecogs.com/svg.latex?\Large&space;s">), while <img style="height: 15px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)-V_\phi(s_t)"> tells us how valuable it is. In other terms <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_\theta (a_t|s_t)"> is the actor, <img style="height: 15px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)-V_\phi(s_t)"> is the critic.



Computation of the Critic can have different flavors :

- Q Actor-Critic (two networks)
- **Advantage Actor-Critic**
- TD Actor-Critic
- TD(<img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda">) Actor-Critic
- Natural Actor-Critic



In my implementation I used advantage Actor Critic. This is basically the synchronous implementation of A3C. A2C is like A3C but without the asynchronous part; this means a single-worker variant of the A3C. It was empirically found that A2C produces comparable performance to A3C while being more efficient. On each learning step, we update both the Actor parameter (with policy gradients and advantage value), and the Critic parameter (with minimizing the mean squared error with the Bellman update equation). 

# Deep Deterministic Policy Gradient

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. This approach is closely connected to Q-learning, and is motivated the same way: if you know the optimal action-value function <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(s,a)">, then in any given state, the optimal action <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;a^*(s)"> can be found by solving

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;a^*(s) = \mathop{\mathrm{argmax}}_aQ^*(s,a)">


#### Quick Facts

- DDPG is an off-policy algorithm.
- DDPG can only be used for environments with continuous action spaces.
- DDPG can be thought of as being deep Q-learning for continuous action spaces.



#### The Q-Learning Side

The Bellman Equation <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(s,a)"> describes thr optimal action-value function:

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;Q^*(s,a) = \mathop{\mathrm{E}}_{s'\sim P} \bigg [ r(s,a) + \gamma \max_{a'} Q^*(s',a') \bigg ]">



Suppose the approximator is a neural network <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q_\phi(s,a)">, with parameters <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\phi">, and that we have collected a set <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{D}"> of transitions <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;(s,a,r,s',d)"> (where <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;d"> say is <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;s'"> is a terminal state). We can set up a **mean-squared Bellman error (MSBE)** function, which tells us roughly how closely <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q_\phi"> comes to satisfying the Bellman equation:

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;L(\phi,\mathcal{D})=\mathop{\mathrm{E}}_{(s,a,r,s',d)\sim \mathcal{D}} \Bigg [ \bigg (Q_\phi(s,a) - \big (r+\gamma (1-d)\max_{a'} Q_{\phi}(s',a') \big) \bigg)^2 \Bigg]">


When `d==True`- which is to say, when <img style="height: 12px;" src="https://latex.codecogs.com/svg.latex?\Large&space;s'"> is a terminal state—the Q-function should show that the agent gets no additional rewards after the current state. Q-learning algorithms for function approximators, such as DQN (and all its variants) and DDPG, are largely based on minimizing this MSBE loss function. There are two main tricks employed by all of them:

- **Replay Buffer**: If you only use the very-most recent data, you will overfit to that and things will break; Therefore, we use a replay buffer to save some previous experiences
- **Target Networks.** Q-learning algorithms make use of target networks. The term <img style="height: 20px;" src="https://latex.codecogs.com/svg.latex?\Large&space;r+\gamma (1-d)\max_{a'} Q_{\phi}(s',a')"> is called the target, because when we minimize the MSBE loss, we are trying to make the Q-function be more like this target. Problematically, the target depends on the same parameters we are trying to train: <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\theta">. This makes MSBE minimization unstable. The solution is to use a set of parameters which comes close to <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\phi">, but with a time delay—that is to say, a second network, called the target network, which lags the first.



#### The Policy Learning Side of DDPG

Policy learning in DDPG is fairly simple. We want to learn a deterministic policy <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;\mu_\theta(s)"> which gives the action that maximizes <img style="height: 14px;" src="https://latex.codecogs.com/svg.latex?\Large&space;Q_\phi(s,a)">. Because the action space is continuous, and we assume the Q-function is differentiable with respect to action, we can just perform gradient ascent (with respect to policy parameters only) to solve

<img  src="https://latex.codecogs.com/svg.latex?\Large&space;\max_\theta \mathop{\mathrm{E}}_{s \sim \mathcal{D}} [Q_\phi(s,\mu_\theta(s))]">



#### Exploration ca. Exploitation

DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their actions at training time. The authors of the original DDPG paper recommended time-correlated [OU noise](https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process), but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of training



#### Pseudocode

![](https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg)




Sources:

1. [https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)
2. [https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)
3. [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. [https://towardsdatascience.com/introduction-to-actor-critic-7642bdb2b3d2](https://towardsdatascience.com/introduction-to-actor-critic-7642bdb2b3d2)
5. [https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
6. [https://spinningup.openai.com/en/latest/algorithms/ddpg.html](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

