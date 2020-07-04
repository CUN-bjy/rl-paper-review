[이전 페이지](README.md)

### 2. DPG

[Deterministic policy gradient algorithms]

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014).

[paper_link](http://proceedings.mlr.press/v32/silver14.pdf)



#### [Abstract]

- 해당 논문에서는 continuous action을 다루는 RL을 위한, ***deterministic* policy gradient algorithm**을 다룬다. 
  - 이는 action-value function의 expected gradient의 형태로 나타나며, 일반적인 stochastic PG에서보다도 효과적으로 추정될 수 있다.

- 또한, 적절한 탐험(새로운 states 경험)을 보장하기 위해 **off-policy actor-critic 방법**을 소개한다.

- 마지막으로, **high-demensional action spaces**에서 stochastic 방식보다 **deterministic PG가 훨씬 뛰어남을 증명**한다.



#### [Introduction]

Policy Gradient 알고리즘은 일반적으로 stochastic한 정책을 통해 샘플링을 한 후 보다 나은 reward를 얻을 수 있는 방향으로 정책파라미터를 조정한다.

반면, 이 논문에서는 deterministic policy를 다루며, 마찬가지로 policy gradient방향으로 정책 파라미터를 조절한다.
$$
\pi_\theta(a|s) = P[a|s;\theta]  \qquad\qquad\qquad\quad a = \mu_\theta(s)
\\ \qquad\quad[\text{stochastic policy}] \qquad\quad\qquad [\text{deterministic policy}]
$$
deterministic policy gradient는 간단한 model-free의 형태로 존재하는데, 이는 action-value function의 gradient를 따른다.

또한, 이 논문에서는  deterministic policy gradient가 stochastic policy gradient의 특별한 케이스(policy의 varience가 0에 수렴)임을 보일 것이다.



stochastic 과 determisitic policy 사이에 결정적인 차이가 존재하는데,

stochastic의 경우에는  policy gradient가 **state와 action spaces 모두와 연관**되어있는 반면에, deterministic의 경우는 **오직 state space와 연관**되어있다는 것이다.

그 결과, stochastic policy gradient를 구하기 위해서는 보다 더 많은 states 샘플을 요구하게 될 것이다. 특히 high-demensional action spaces의 경우에서 말이다.



모든 state와 action을 탐험하기 위해서는, stochastic policy방법이 필수적이다.

이 논문에서는 deterministic policy gradient알고리즘이 충분히 탐험(새로운 states를 경험)을 지속할 수 있도록 **off-policy learning algorithm**을 제시한다.

이 알고리즘의 기본적인 아이디어는 **stochastic behaviour policy에 따라 action을 선택**하되, **deterministic target policy를 이용해 학습**을 한다는 것이다.



이 논문에서는 **off-policy actor-critic algorithm**을 유도하기위해 deterministic policy gradient를 사용한다. 

여기서 off-policy actor-critic algorithm은 미분가능한 근사함수를 이용해 action-value function을 추정하고, 

근사된 action-value gradient의 방향으로 policy parameter를 업데이트 한다. 



또한, deterministic policy gradient를 근사하기위해 SPG와 마찬가지로(내용은 다르겠지만) **compatible function**을 소개한다.

이는 policy gradient가 biased-estimated 되지 않았음을 보이기 위함이다.



#### [Background]

##### Preliminaries

##### Stochastic Policy Gradient Theorem

##### Stochastic Actor-Critic Algorithms

##### Off-Policy Actor-Critic



#### [Gradients of Deterministics Policies]

##### Action-Value Gradients

##### Deterministic Policy Gradient Theorem



#### [Deterministic Actor-Critic Algorithms]

##### On-Policy Deterministic Actor-Critic

##### Off-Policy Deterministic Actor-Critic

##### Compatible Function Approximation

