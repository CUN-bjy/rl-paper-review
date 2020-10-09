[HOME](../README.md)

### 4. NPG

[A natural policy gradient]

Sham Kakade(2002)

[paper_link](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)

</br>

### [Abstract]

- 이 논문에서는 **natural gradient method**를 제안하며, 이는 학습할 parameter의 steepest descent direction을 구할 수 있게 해준다.
- gradient method가 parameter에 큰 변화를 줄 수는 없지만, natural gradient method가 조금 더 나은 action을 선택하기보다는 **greedy-optimal action**을 선택하는 방향으로 나아감을 보인다.
- 이러한 greedy-optimal action은 Sutton이 제안한 compatible value function을 근사하는 과정에서 선택되는 거소가 같으며, 이때 간단한 MDP와 복잡한 MDP(e.g. Tetris)에서 drastic performance improvements을 나타냄을 보인다.

</br>

### [Introduction]

direct-policy-gradient 방법으로 복잡한 MDP문제를 해결하기위한 관심들은 계속 증가되어왔다.

이러한 방식들은 대개 future reward의 gradient를 따라 몇가지 제한된 policies의 종류 중에 좋은 policy를 고른다.

</br>

안타깝게도, 전형적인 gradient descent rule은 **non-covariant**하다.

그 말은 즉, 아래 식에서 <u>좌변에는 theta, 우변에는 1/theta를 각각 가지기 때문에 inconsistent하다</u>는 것이다.(?)
$$
\Delta\theta_i=\alpha\frac{\partial f}{\partial \theta_i}
$$
</br>

이 논문에서는 그에대한 대안으로  **natural gradient**라는 이름의 covariant gradient를 제시한다.

또한 natural gradient가 greedy-optimal-action을 선택하는 방향으로 움직인다는 것을 보인다.

이 방법을 사용하여 local minimum에 빠지는 plateau 현상과 같은것은 이제 심각한 문제가 아님을 보인다.

</br>

### [A Natural Gradient]

논문에서는 기본적인 설명을 위한 기본 정의 및 표현들을 열거하며 전개해나간다.
$$
\text{finite MDP : } (S,s_0,A,R,P)
\\S : \text{finite set of states}, \\s_0 : \text{start state},\\A : \text{finite set of actions}, \\R : S \times A \to [0,R_{max}] \text{, reward function}, \\P : \text{transition model}\\ policy \to\pi(a;s), \text{stochastic policy}
$$
모든 policy는 확률적으로 고르게 분포되어있다 가정하며, 즉, 잘 정의된 stationary distribution을 가지고 있다.

이러한 가정 속에서 또한 다음의 식을 만족한다.
$$
\text{average reward}: \eta(\pi) \equiv \sum_{s,a}\rho^\pi(s)\pi(a;s)R(s,a)
\\\text{state-action value}: Q^\pi(s,a) \equiv E_{\pi}[\sum_{t=0}^\infty R(s_t,a_t)-\eta(\pi)|s_0=s,a_0=a]\\\text{value function}:J^\pi(s)\equiv E_{\pi(a^\prime;s)}[Q^\pi(s,a^\prime)]
$$

이어서 이 논문의 목표인 average reward를 최대화해주기위한 policy를 찾기위해  parameterized policy를 다음과 같이 표현할 수 있으며, 
$$
\Pi = [\pi_\theta:\theta \in \mathfrak{R}^m],
\\
$$
이를 이용한 average reward의 gradient를 구하는 식은 다음과 같다.
$$
\nabla\eta(\theta) = \sum_{s,a}\rho^\pi(s)\nabla\pi(a;s,\theta)Q^\pi(s,a)
$$

</br>

일반적으로 average reward의 steepest descent direction는 아래와 같이 정의된다.
$$
\text{steepest descent direction : }d\theta \text{ that minimizes }\eta(\theta + d\theta)
\\ \text{under the constraint}(|d\theta|^2 \to \text{a small constant})
$$
이 때, squared length는 일종의 **positive-definite matrix**와 함께 정의된다.
$$
|d\theta|^2 \equiv \sum_{ij}G_{ij}(\theta)d\theta_id\theta_j = d\theta^T G(\theta)d\theta\text{(using vector notation)}
$$






### [The Natural Gradient and Policy Iteration]

#### Compatible Function Approximation



#### Greedy Policy Improvement



### [Metrics and Curvatures]



### [Experiments]



### [Discussion]