---
title: "Deep RL Bootcamp - Hackathon"
description: "Results from Pascal Sager"
permalink: /conclusion
---

# Conclusion


## Possible Improvements and next Steps
I investigated the environment at the beginning and decided to use model-free RL algorithms. These can be divided into three categories:
- Q-Learning
- Policy Optimization
- Hybrid between Q-Learning and Policy-Optimization

I was able to complete one Q-learning algorithm (DQN) and two policy algorithms (A2C, PPO). I started with a hybrid of both methods (DDPG), but did not finish in time.

### DQN
DQN worked very well. This is due to the sample efficiency of the algorithm. But the result was a bit unstable. However, this is also due to the fact that I deliberately optimized to achieve the 200 points as fast as possible. Therefore I have been willing to accept this instability. In an industrial application, however, this would probably not be optimal and it should rather be optimized for a somewhat slower but more stable solution.
I have done some experiments for DQN. However, I could not tune the hyperparameters in this short time. I think if sweeps were done, the result could be further improved. Additional parameters like the batch size should also be considered in a further tuning.

### PPO
After implementing DQN, I implemented the first policy optimization algorithm with PPO. It learns the policy directly and is therefore more stable but less sample efficient. This has been reflected in the results: The reward increases slowly but constantly. Even over a very long period of 4.5h the reward remains constant. As with DQN, sweeps should also be performed for PPO.

### A2C
PPO achieved a very stable result, but it took much more time than DQN. I did not know if this is due to the algorithm, the implementation or if this is a "property" of Policy Optimization algorithms. Policy algorithms tend to be very stable but less sample efficient. Since I could not optimize PPO to reach the 200 points faster I tried A2C.

I was a little lucky with A2C. Under normal circumstances, A2C was also very stable. But since it doesn't have clipping like PPO, it can become unstable more easily. Due to this instability, certain parameters let the reward fast increase to a reward >200 points (for a few hundred episodes). This reward could then not be kept over a longer time like with PPO. Thus I could show that policy optimization methods can also win the game quickly. In the "normal" setup, however, A2C is also very stable and then has a little longer to win the game.

## Personal Conclusion / Feeling
Personally, I think that I have achieved a very good result in this short time. Not only did I win the game, but I also 
achieved the 200 points in a very good time. Nevertheless, there is of course a lot of potential for improvement. However,
I think that this is normal for a hackathon. No perfect solution can be expected within 56h. From a software point of view,
the structure should certainly be improved. Also, much more experimentation could be done to further improve the performance.
Overall, I think my procedure was very good. I approached the problems in a very structured way and achieved good results.
However, this hackathon also showed me my limits. I worked at least 15h+ every day and only took about 1.5h break. This 
hackathon setup was pushing me as I wanted to get a good result. I think I succeeded and I am happy about that.




