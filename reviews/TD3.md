[HOME](../README.md)

### TD3

[Addressing Function Approximation Error in Actor-Critic Methods]

Scott Fujimoto , Herke van Hoof , David Meger (2018)

[`PAPER`](https://arxiv.org/pdf/1802.09477.pdf)

<br/>

### *What is differences from DDPG*

1. Overestimation Bias Problem Solved
   - double Q-network 

   - clipped double q-update

2. Addressing Variance
   - delayed policy update
   - target policy smoothing