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
    background-image: url("https://lh3.googleusercontent.com/proxy/x04EA4RhDz_One5GheU5HN7NR-NjI6mR5rDj8SW2CgiHd83RIi_Kl6bXM5VPh8USIqUgDYIgU388XIOEmR9wfJU9sYLBW9w4SYgbp2z-7KcePihTmQ");
    background-repeat: no-repeat;
    background-position: right;
    background-size: 14%;
}
section {
    padding-top: 120px;
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
- Selection of the Algorithms
- Brief Explanation of the Algorithms
- Results
- Performance
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
Selection of the Algorithms
</div>

- **Inverse RL** was not considered, because the reward function is defined by the environment
- **Model-Based vs. Model-Free**: Has the agent access to (or learns) a model of the environment (model = function which predicts state transitions and rewards)
$\rightarrow$ **model free algorithms used**
- **What to learn**:
  - **Policy Optimization** (model free, on-policy): Directly optimize the policy, makes algorithm more stable and reliable (see [spinningup.openai.com/](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html))
  - **Q-Learning** (model free, off-policy): Learn approximator for the optimal action-value function, update can use data collected at any point during training, more sample efficient

---

<div class=custom-header>
Selection of the Algorithms
</div>

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)


---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>

#### Deep Q-Learning (Double Q-Learning)
- Collect rollouts in a replay buffer $\rightarrow$ approximate the Q-Value function $Q(S,A)$
- Prediction Network executes steps in environment, Target network calculates gradients on "frozen" parameters

#### Clip Proximal Policy Optimization
- Tries to make the biggest possible policy improvment step without causing a performance collapse
- Two policy networks: one to refine the policy and one to collect samples $\rightarrow$ networks are synchronized after a specific intervall


---

<div class=custom-header>
Brief Explanation of the Algorithms
</div>

#### Deep Deterministic Policy Gradient
> TODO überarbeiten
- Learns concurrently a Q-function and a policy
- Uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy

#### Advantage Actor-Critic
> TODO überarbeiten
- Actor: Performs the action in the environment
- Critic: Estimates how valuable a specific action is


---

<div class=custom-header>
Results
</div>

TODO

---

<div class=custom-header>
Performance
</div>

TODO

---

<div class=custom-header>
Conclusion and Outlook
</div>

TODO

