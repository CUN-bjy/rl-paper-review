[이전 페이지](README.md)

### 2. DPG

[Deterministic policy gradient algorithms]

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014).

[paper_link](http://proceedings.mlr.press/v32/silver14.pdf)



#### [Abstract]

해당 논문에서는 continuous action을 다루는 RL을 위한, *deterministic* policy gradient algorithm을 다룬다.

이는 action-value function의 expected gradient의 형태로 나타나며, 일반적인 stochastic PG보다 효과적으로 추정된다.

또한, 적절한 탐험(새로운 states 경험)을 보장하기 위해 off-policy actor-critic 방법을 소개한다.

마지막으로, high-demensional action spaces에서 stochastic 방식보다 deterministic PG가 훨씬 뛰어남을 증명한다.



#### [Introduction]