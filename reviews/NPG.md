[HOME](../README.md)

### 4. NPG

[A natural policy gradient]

Sham Kakade(2002)

[paper_link](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)



### [Abstract]

- 이 논문에서는 **natural gradient method**를 제안하며, 이는 학습할 parameter의 steepest descent direction을 구할 수 있게 해준다.
- gradient method가 parameter에 큰 변화를 줄 수는 없지만, natural gradient method가 조금 더 나은 action을 선택하기보다는 **greedy-optimal action**을 선택하는 방향으로 나아감을 보인다.
- 이러한 greedy-optimal action은 Sutton이 제안한 compatible value function을 근사하는 과정에서 선택되는 거소가 같으며, 이때 간단한 MDP에서는 drastic performance improvements을 나타냄을 보인다.



### [Introduction]



### [A Natural Gradient]



### [The Natural Gradient and Policy Iteration]

#### Compatible Function Approximation



#### Greedy Policy Improvement



### [Metrics and Curvatures]



### [Experiments]



### [Discussion]