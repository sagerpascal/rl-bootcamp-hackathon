---
marp: true
theme: default
footer: Deep RL Bootcamp - Hackathon
paginate: true
---

<!-- Global style -->

<style>

.custom-header {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    padding: 5px 70px;
    background-color: #0064A6;
    color: white;
    font-size: 1.6em;
    background-image: url("https://lh3.googleusercontent.com/proxy/IY_lk6EmHAdYnuRwXYzOQMbDB-pz8Kh7cxczaMVEfSm7Y9vh5cWQTX106cJN2lZ8hG4dr5k9AsCBCfzRimp13FthcV7kJg-wanB7uxok8uiJ8Ydxig");
    background-repeat: no-repeat;
    background-position: right;
    background-size: 14%;
}
section {
    padding-top: 120px;
}

.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
}

</style>

# Deep RL Bootcamp - Hackathon
### Results from Pascal Sager

January 15, 2021



---
<div class=custom-header>
Content
</div>

- Documentation
- Environment
- Procedure
- Selection of the Algorithms
- Brief Explanation of the Algorithms
- Results
- Conclusion and Outlook

---
<div class=custom-header>
Documentation
</div>

I have created a **detailed documentation on Github Pages**: [https://sagerpascal.github.io/rl-bootcamp-hackathon/](https://sagerpascal.github.io/rl-bootcamp-hackathon/)


<br>

*This presentation is only a summary of the documentation.*


---

<div class=custom-header>
Environment
</div>

### LunarLander-V2
- Land between the two flags

<img src="https://user-images.githubusercontent.com/48079888/93164280-23663300-f6e7-11ea-8cf9-e049d887a4ca.gif" style="width:500px; height:300px" class="center">

- **Points**: Speed, landing position, usage of engine

---

<div class=custom-header>
Procedure
</div>

**Approach:**

1. Setup docker on the GPU-cluster
2. Select the algorithm
3. Implement simple basic version
4. Compare with high-end implementations and improve the algorithm
5. Tuning and documentation
6. Repeat from 2.

**Principles:**
- Quality before quantity: Good performance over many different algorithm
- **Work like hell :)**


---

<div class=custom-header>
Selection of the Algorithms
</div>

- **Inverse RL** was not considered, because the reward function is well defined by the environment
- **Model-Based vs. Model-Free**: Has the agent access to (or learns) a model of the environment (model = function which predicts state transitions and rewards)
$\rightarrow$ **model free algorithms used**
- **What to learn**:
  - **Policy Optimization** (model free, on-policy): Directly optimize the policy, makes algorithm more stable and reliable (see [spinningup.openai.com/](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html))
  - **Q-Learning** (model free, off-policy): Learn approximator for the optimal action-value function, update can use data collected at any point during training, more sample efficient

---

<div class=custom-header>
Selection of the Algorithms
</div>


<img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg"  class="center">

---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>

#### Double Deep Q-Network (Double DQN)
- Collect rollouts in a replay buffer $\rightarrow$ approximate the Q-Value function $Q(S,A)$


<img src="\rl-bootcamp-hackathon\assets\images\presentation\alg_dqn.PNG" alt="algorithm" style="height:88%; width:60%" class="center"/>

---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>

#### Clip Proximal Policy Optimization (Clip PPO2)
- Use on-policy value function $V^{\pi}(s)$ to figure out how to update the policy $\pi_{\theta}(a|s)$

<img src="\rl-bootcamp-hackathon\assets\images\presentation\alg_ppo.PNG" alt="algorithm" style="height:88%; width:65%" class="center"/>

---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>

#### Advantage Actor-Critic (A2C)

- Similar to PPO, but without clipping and with policy entropy

<img src="\rl-bootcamp-hackathon\assets\images\presentation\alg_a2c.PNG" alt="algorithm" style="height:88%; width:75%" class="center"/>

---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>


#### Deep Deterministic Policy Gradient (DDPG)

  - Collect rollouts (as for DQN)
  - **Two prediction networks**: Actor and Critic (as for Q-Actor-Critic)
  - **Two target networks**: Target-Actor and Target-Critic (as for DQN)
  - Learns concurrently a Q-function and a policy
  - Uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy

<br>

**Algorithm not finished yet!**

---

<div class=custom-header>
Results
</div>

[https://sagerpascal.github.io/rl-bootcamp-hackathon/results](https://sagerpascal.github.io/rl-bootcamp-hackathon/results)

---


<div class=custom-header>
Conclusion and Outlook
</div>

### Algorithm
- **DQN**: Very sample efficient but less stable, reached a pretty high avgerage score
- **PPO**: Very stable, took longer to win the game
- **A2C**: The proof that Policy Optimization can also win fast (with a little luck)

### Next Steps
- Finish DDPG and win Lunar-Landing with continous action space
- Cleanup code
- Run sweeps
- ...

---

# Discussion