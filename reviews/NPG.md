[HOME](../README.md)

### 4. NPG

[A natural policy gradient]

Sham Kakade(2002)

[paper_link](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)

</br>

같이 읽으면 도움이 되는 글들

- https://www.slideshare.net/SooyoungMoon3/natural-policy-gradient

</br>

### [0. Abstract]

- 이 논문에서는 **natural gradient method**를 제안하며, 이는 학습할 parameter의 steepest descent direction을 구할 수 있게 해준다.
- gradient method가 parameter에 큰 변화를 줄 수는 없지만, natural gradient method가 조금 더 나은 action을 선택하기보다는 **greedy-optimal action**을 선택하는 방향으로 나아감을 보인다.
- 이러한 greedy-optimal action은 Sutton이 제안한 compatible value function을 근사하는 과정에서 선택되는 것과 같으며, 이때 간단한 MDP와 복잡한 MDP(e.g. Tetris)에서 drastic performance improvements을 나타냄을 보인다.

</br>

### [1. Introduction]

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

### [2. A Natural Gradient]

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
이 때, squared length는 일종의 **positive-definite matrix**와 함께 표현되며,
$$
|d\theta|^2 \equiv \sum_{ij}G_{ij}(\theta)d\theta_id\theta_j = d\theta^T G(\theta)d\theta\text{(using vector notation)}
$$

위 정의에 따라 계산해보면, steepest descent direction을 아래와 같이 간단히 나타낼 수 있다.
$$
\text{steepest descent direction : }G^{-1}\nabla\eta(\theta)
$$
일반적인 gradient descent는 위 수식의 함수 **G**를 identity matrix **I**로 가정한 

특수 케이스의 steepest descent direction을 따르 것인데,

coordinates기반의 metric이 아닌 <u>manifold기반의 metric</u>으로 일반화 한 것이 바로 위의 결과이다.

그리고 이러한 metric을 바로 **Natural Gradient**라고 부른다.

</br>

이 정책 파라미터의 **Fisher information matrix**는 다음과 같이 정의될 수 있는데.
$$
F_s(\theta) \equiv E_{\pi(a;s,\theta)}[\frac{\partial log\pi(a;s,\theta)}{\partial \theta_i}\frac{\partial log\pi(a;s,\theta)}{\partial \theta_j}]
$$
이는 분명 **positive-definite**하다.

 또한 모든 coordinates에 대해 *same distance*를 지닌다는 이유에서 **invariant** 하다.

</br>

Fisher information은 probability manifold에서 그에 해당하는 state s에서의 <u>distance</u>를 측정해주는 도구이다.

average reward가 정책 파라미터로부터 정의되었다는 점에서

정책 파라미터의 straightforward를 측정하는 도구를 다음과 같이 나타낼 수 있다. 
$$
F(\theta) = E_{\rho^\pi(s)}[(\frac{\partial}{\partial\theta}log\pi(a;s,\theta))^2]=E_{\rho^\pi(s)}[\frac{\partial log\pi(a;s,\theta)}{\partial \theta_i}\frac{\partial log\pi(a;s,\theta)}{\partial \theta_j}] = E_{\rho^\pi(s)}[F_s(\theta)]
$$
직관적으는, 매 state의 거리 평균을 계산해 정책의 진행방향(straightforward)을 수치적으로 계산한다고 할 수 있겠다.

이처럼, 모든 coordinates에 대해 **invariant**하며 **straightforward metric**으로서 

positive-definite함수 G를 대체할 수 있는 함수 Fisher Information matrix를 이용한다면,

아래와 같이 steepest descent direction을 나타낼 수 있다.
$$
\overset{\sim}\nabla\eta(\theta) \equiv F(\theta)^{-1}\nabla\eta(\theta)
$$
</br>


### [3. The Natural Gradient and Policy Iteration]

이번 장에서는 natural gradient를 사용한 policy improvement와 일반적인 policy iteration을 비교할 것이다.

보다 적절한 비교를 위해, action-value funtion Q를 parameter w로 이루어진 *compatible* function approximator로 근사할 것이다.

#### 3.1 Compatible Function Approximation

$$
\text{For vectors }\theta, w \in \R^m,
\\ \psi(s,a)^\pi = \nabla log\pi(a;s,\theta),\quad f^\pi(s,a;w) = w^T\psi^\pi(s,a)
\\ \text{where} [\nabla log\pi(a;s,\theta)]_i = \partial log\pi(a;s,\theta)/\nabla\theta_i
$$

compatible value function f가 위와같이 parameter w에 의해 정의될 때.

squared error는 다음과 같으며,
$$
\epsilon(w,\pi) \equiv \sum_{s,a}\rho^\pi(s)\pi(a;s,\theta)(f^\pi(s,a;w)-Q^\pi(s,a))^2
$$
이때 compatible function f를 true value 대신 사용했을 때 gradient를 계산할 수 있다는 점에서 function approximator가 <u>policy와 compatible</u>하다고 할 수 있다.

</br>

***Theorem 1.***

squared error가 최소화되었을 때의 파라미터 w는  steepest descent direction과 같다.
$$
\overset{\sim}w = \overset{\sim}\nabla\eta(\theta)
$$
*Proof.*(자세한건 논문참조)
$$
\text{when }\overset{\sim}w \text{ minimizes the squared error, }\partial\epsilon/\partial w_i = 0,
\\ \sum_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)(\psi^\pi(s,a)^T\overset{\sim}w-Q^\pi(s,a)) = 0 ,
\\ (\sum_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)\psi^\pi(s,a)^T)\overset{\sim}w = \sum_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)Q^\pi(s,a)
\\ F(\theta)\overset{\sim}w = \nabla\eta(\theta), \text{by definition of }\psi^\pi
\\\therefore \overset{\sim}w = F^{-1}(\theta)\nabla\eta(\theta) = \overset{\sim}\nabla\eta(\theta) \quad \Box.
$$

즉, actor-critic framework는 function approximator의 weight를 natural gradient로서 사용하고 있었던 것.

</br>


#### 3.2 Greedy Policy Improvement

greedy policy improvement단계는 approximator를 이용해 각 state에서 높은 value를 가지는 action을 선택하는 것이다. 이번 절에서는 natural gradient가 단지 좋은 action이 아니라 *best action*으로 향하도록 한다는 것을 보일 것이다.

</br>

우선, exponential 함수의 형태의 정책을 고려해보자.
$$
\pi(a;s,\theta) \propto exp(\theta^T\phi_{sa}), \text{where } \phi_{sa} \text{is some feature vector in }\R^m
$$
exponential 함수를 먼저 고려한 이유는 바로 ***affine geometric***한 성질을 지니고 있기 때문이다.

이는 접선벡터(tangent vector)에의 한 점이 변환되더라도 manifold의 한 점으로 남아있을 수 있도록 한다.

</br>

일반적으로 정책 파라미터의 probability manifold는 curve형태이며, 

법선벡터에 의해 한 점이 변환된다면 그 점이 manifold공간(예컨데 sphere 공간)에서 유지되리란 보장이 없다.

후에는 일반적인 형태(non-exponential)에 대해서 고려해보아야겠지만, 지금은 exponential case를 먼저 보도록하자.

</br>

이번에는 exponential 함수형태의 정책이 natural gradient 방향으로의 충분히 큰 스텝(learning rate를 아주 크게)으로 학습한다면 greedy policy improvement step를 통해 도달할 수 있는 정책과 동일함을 보이려 한다.

***Theorem 2.***
$$
\text{For }\pi(a;s,\theta) \propto exp(\theta^T\phi_{sa}),
\\\text{ assume } \overset{\sim}\nabla\eta(\theta) : \text{non-zero, and } 
\overset{\sim}w \text{ minimizes approximation error.}
\\Let \quad\pi_\infin(a;s) = lim_{a\to\infin}\pi(a;s,\theta+\alpha\overset{\sim}\nabla\eta(\theta)).
\\Then \quad\pi_\infin(a;s) \ne \text{if and only if }a \in argmax_{a^\prime}f^\pi(s,a\prime,\overset{\sim}w)
$$
*Proof.*(자세한건 논문참조)

compatible function approximator에 theorem의 결과를 적용하면,
$$
f^\pi(s,a;\overset{\sim}w) = \overset{\sim}\nabla\eta(\theta)^T\psi^\pi(s,a)
$$
또한,  정책함수의 정의에 따라,
$$
\pi(a;s,\theta) \propto exp(\theta^T\phi_{sa}),\\
\psi^\pi(s,a) = \nabla log\pi(a;s,\theta) =\phi_{sa} - E_{\pi(a^\prime;s,\theta)}(\phi_{sa^\prime})
$$

이를 합쳐보면,
$$
f^\pi(s,a;\overset{\sim}w) = \overset{\sim}\nabla\eta(\theta)^T[\phi_{sa}-E_{\pi(a^\prime;s,\theta)}\phi_{sa^\prime}]
$$
여기서 Expectation term은 a에 대한 함수가 아니므로 다음을 따른다.(?)
$$
argmax_{a^\prime}f^\pi(s,a^\prime;\overset{\sim}w) = argmax_{a^\prime}\overset{\sim}\nabla\eta(\theta)^T\phi_{sa^\prime}
$$





### [4. Metrics and Curvatures]



### [5. Experiments]



### [6. Discussion]