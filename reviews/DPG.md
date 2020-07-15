[HOME](../README.md)

### 2. DPG

[Deterministic policy gradient algorithms]

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014).

[paper_link](http://proceedings.mlr.press/v32/silver14.pdf)



### [Abstract]

- 해당 논문에서는 continuous action을 다루는 RL을 위한, ***deterministic* policy gradient algorithm**을 다룬다. 
  - 이는 action-value function의 expected gradient의 형태로 나타나며, 일반적인 stochastic PG에서보다도 효과적으로 추정될 수 있다.

- 또한, 적절한 탐험(새로운 states 경험)을 보장하기 위해 **off-policy actor-critic 방법**을 소개한다.

- 마지막으로, **high-demensional action spaces**에서 stochastic 방식보다 **deterministic PG가 훨씬 뛰어남을 증명**한다.



### [Introduction]

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



### [Background]

#### Preliminaries

일반적인 stochastic policy가 정의된 MDP에서 우리는 아래와 같이 **performance objective**를 정의할 수 있으며,

expectation의 형태로 나타낼 수 있다. (변수설명은 논문 참조)
$$
J(\pi_\theta) = \int_S \rho^\pi(s)\int_A \pi_\theta(s,a)r(s,a)dads
\\ = E_{s\sim p^\pi,a\sim\pi^\theta}[r(s,a)]\qquad\quad
$$

#### Stochastic Policy Gradient Theorem

**Policy Gradient** algorithms은 continuous action reinforcement learning algorithm에서 가장 유명한 알고리즘일 것이다.

이 알고리즘의 주된 아이디어는 정책파라미터인 theta를 아래의 **performance gradient**의 방향으로 조정하는 것이다. 
$$
\nabla_\theta J(\pi_\theta) = \int_S\rho^\pi(s)\int_A\nabla_\theta\pi_\theta(a|s)Q^\pi(s,a)dads
\\ \quad\quad=E_{s\sim\rho^\pi,a\sim\pi_\theta}[\nabla_\theta log\pi_\theta(a|s)Q^\pi(s,a)]
$$
policy gradient알고리즘은 놀라울정도로 간단하다.

state distribution은 정책파라미터에 연관성이 있는데도 불구하고, state distribution의 gradient와 policy gradient는 서로 무관하다.

위 사실은 performance gradient를 계산할 필요없이 sampling을 통해 expectation을 구할 수 있도록 도와주었다.

이 덕분에 다양한 policy gradient algorithm들이 유도되기 시작하였고, 

이러한 알고리즘들을 다루기 위해서는 **action-value funciton을 어떻게 추정할 것인가**에 대한 문제로 귀결되었다.



#### Stochastic Actor-Critic Algorithms

**actor-critic** 알고리즘은 policy gradient기반의 구조에서 가장 널리 쓰이며, actor와 critic 두가지 요소로 이루어져있다.



**actor**는 performance gradient의 stochastic gradient ascent를 이용해 stochastic policy를 조절하며, 

알려지지 않은 action-value function을 파라미터로 근사해  대체한다.
$$
\nabla_\theta J(\pi_\theta) = E_{s\sim\rho^\pi,a\sim\pi_\theta}[\nabla_\theta log\pi_\theta(a|s)Q^w(s,a)]
$$
**critic**은 action-value function을 적절한 정책 평가 알고리즘(e.g. temporal-difference learning)으로 추정한다.

일반적으로 action-value function을 function approximator로 대체하는 것은 bias하다고 알려져있으나,

function approximator가 **compatible** 하다면 bias하지 않다.



여기서 compatible한 funciton approximator의 조건은 다음과 같다.
$$
\text{i)}\quad Q^w(s,a) = \nabla_\theta log\pi_\theta(a|s)^Tw
\\ \text{ii)} \quad \text{parameter } w\text{ are chosen to minimize the mean-squared error} 
\\ \epsilon^2(w) = E_{s\sim \rho^\pi,a\sim\pi_\theta}[(Q^w(s,a)-Q^\pi(s,a))^2]
$$


#### Off-Policy Actor-Critic

별도의 behaviour policy로부터 trajectories를 sampling하는 **off-policy** 방식의 policy gradient algorithm은 종종 유용하게 사용된다.
$$
\beta(a|s) \neq \pi_\theta(a|s)
$$
일반적인 off-policy 방법에서, **performance objective**는 아래의 식과 같이 

behaviour policy의 state distribution에 대해 averaged된 target policy의 value function으로 수정된다.(???한국어로 어떻게 번역을해야할까)

(참고: modified to be the value function of ther target policy, averaged over the state distribution of the behaviour policy)
$$
J_\beta(\pi_\theta) = \int_S\rho^\beta(s)V^\pi(s)ds
\\ \qquad\qquad\qquad\qquad\quad = \int_S\int_A\rho^\beta(s)\pi_\theta(a|s)Q^\pi(s,a)dads
$$
또, 이에 대해 미분된 performance objective는 **off-policy policy-gradient**로 근사된다.
$$
\nabla_\theta J_\beta(\pi_\theta) \approx \int_S\int_A\rho^\beta(s)\nabla_\theta\pi_\theta(a|s)Q^\pi(s,a)dads
\\ \qquad\qquad\qquad = E_{s\sim\rho^\beta,a\sim\beta}[\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}\nabla_\theta log\pi_\theta(a|s)Q^\pi(s,a)]
$$
위 식의 근사는 [(Degris 2012b)](https://arxiv.org/abs/1205.4839) 이 논문에 근거한 근사이며, action-value gradient과 연관된 term이 제거된 것이다.

위 논문에 의하면 이러한 approximation이 gradient ascent가 수렴하는 방향으로 local optima가 형성되므로 충분히 좋은 근사라고 주장한다.



위 논문에서 소개된, **Off-Policy Actor-Critic(OffPAC)**은 behaviour policy를 사용해 trajectories sample을 생성한다.

**critic**은 이 때 생성된 trajectories로부터 off-policy로 state-value function을 추정해내며,

이때 gradient temporal-difference learning을 사용한다.

또한, **actor**는 trajectories로부터 off-policy로 target policy의 파라미터를 업데이트 시킨다.

이때는 stochastic gradient ascent를 이용해 업데이트한다.



여기서 actor와 critic은 behaviour policy가 아닌 target policy를 사용했다는 것을 반영하기위해,

**importance sampling ratio**를 사용한다.
$$
\text{importance sampling ratio : } \space\frac{\pi_\theta(a|s)}{\beta_\theta(a|s)}
$$


### [Gradients of Deterministics Policies]

이 장에서는 어떻게 policy gradient framework이 deterministic policy까지 확장되는가를 보인다.

먼저, deterministic policy gradient에 담긴 이론적인 것들을 전달하고, 이를 증명한다.

마지막으로, deterministic policy gradient theorem이 사실은 stochastic policy gradient theorem의 특수 케이스임을 보인다.



#### Action-Value Gradients

model-free RL algorithm은 주로 일반화된 policy iteration 기법을 기반으로 두고있으며,

이는 **policy evaluation** 과  **policy improvement**로 이루어져있다.



**policy evalution** 방법은 Monte-Carlo evaluation 이나 temporal-difference learning 방법 등으로 action-value function을 추정한다.

**policy improvement** 방법은 추정된 action-value function에 의해 policy가 업데이트 되며, 

주로 action-value function의 greedy maximization방법으로 업데이트한다.



하지만 continuous action spaces에서는 이러한 방식의 policy improvement는 매 스텝 global maximization을 요구하기에 문제가 많았다.

그 대신에 간단하고 계산적으로도 매우 좋아보이는 대안이 있는데, 바로 **Q-function의 gradient 방향으로 policy를 업데이트** 하는 것이다.
$$
\theta^{k+1} = \theta^k + \alpha E_{s\sim\rho^{\mu^k}}p[\nabla_\theta Q^{\mu^k}(s,\mu_\theta(s))]
$$
이러한 방식은 매 state마다 다른 방향으로의 policy improvement가 진행되며, 이들은 state distribution에 대한 기대값으로 취해져 평균치를 이룰 것이다.



또한, chain rule의 적용을 통해 위 식을 action에 대한 action-value function과 policy parameter에 대한 policy gradient로 나눌 수 있다.
$$
\theta^{k+1} = \theta^k + \alpha E_{s\sim\rho^{\mu^k}}[\nabla_\theta\mu_\theta(s)\nabla_a{Q^{\mu^k}(s,a)}|_{a=\mu_\theta(s)}]
$$
하지만, policy의 변화에 의해 방문하게되는 states가 바뀌게 되고 결국  state distribution이 변할것이다. 

이는 결국 distribution에 대한 변화를 모두 설명하지 못하고서는 improvement를 보장할 수 있다기에 불확실함을 가진다고 생각될 것이다. 

그러나, stochastic policy gradient theorem에서 보여진 것과 같이 **state distribuiton의 gradient는 계산할 필요가 없다**.

따라서, 위 식에서 보여진 직관적인(?) 업데이트방식은 분명 performance objective의 gradient를 따르니 걱정하지 말도록하자.



#### Deterministic Policy Gradient Theorem

stochastic policy와 유사한 방법으로 performance objective를 표현하자면 다음의 식과 같다.
$$
J(\mu_\theta) = E[r^\gamma_1|\mu]=\int_S\rho^\mu(s)r(s,\mu_\theta(s))ds
\\ = E_{s\sim\rho^\mu}[r(s,\mu_\theta(s))] \qquad\quad
$$
또한, 해당 performance objective를 지난번에 리뷰한 Sutton의 논문과 유사한 방식으로 policy gradient를 유도해 낸다.

해당 증명은 Appendix에 증명되어 있으며, 그 결과는 아래 식과 같다.



**Theorem 1** Deterministic Policy Gradient Theorem
$$
\nabla_\theta J(\mu_\theta) = \int_S \rho^\mu(s)\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}ds
\\\quad\quad = E_{s\sim\rho^\mu}[\nabla_\theta\mu_\theta(s)\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]
$$
위 식에서도 나타나듯이, expectation값을 구할 때 모든 **state에 대해서만 고려해주면 되는 것이 해당 Theorem의 장점**이다.

stochastic 방식은 state와 action에 대한 기댓값을 구해야 하기 때문에 deterministic방식이 더욱 효율적이라는 것이다.



#### Limit of the Stochastic Policy Gradient 

이 논문에서는 deterministic policy gradient가 사실은 stochastic policy gradient의 스페셜케이스이라는 사실을 보인다.(Appendix C 참고)



deterministic policy로 파라미터화된 stochastic policy가 있다고 가정하자.

그리고 이 policy의 variance parameter가 0에 수렴할 때, stochastic policy와 deterministic policy가 동등하며.

이때 stochastic policy gradient와 deterministic policy gradient 역시 동일하게 수렴한다.



**Theorem 2** 
$$
\text{Consider a stochastic policy } \pi_{\mu_\theta,\sigma} \text{ such that }\pi_{\mu_\theta,\sigma}(a|s) = \nu_\sigma(\mu_\theta(s),a)
\\\lim_{\sigma\to0}\nabla_\theta J(\pi_{\mu_\theta,\sigma}) = \nabla_\theta J(\mu_\theta)
$$

이 사실을 통해 deterministic policy gradient가 stochastic policy gradient의 한 부분이므로,

그동안 policy gradient에 적용되었던 모든 이론적 내용들이 **deterministic policy에도 호환 가능**하다는 사실을 나타낸다. 



### [Deterministic Actor-Critic Algorithms]

이번 절에서는 on-policy와 off-policy actor-critic 알고리즘에 deterministic policy gradient를 사용한다.

첫번째로, 쉬운 예를 들기 위해 간단한 **SARSA critic방법을 이용해 on-policy 업데이트**를 진행하며

다음으로, **Q-learning critic을 사용해 off-policy 업데이트**를 진행해 이 논문의 메인 아이디어를 전달한다.



이러한 알고리즘들은 결국 실질적인 문제점들을 마주치게 된다.

바로 Function Approximator의 **편향성(bias)**과 off-policy learning으로 부터 야기된 **불안정함(high-varience)**이다.

때문에 보다 이론적인 접근으로 **compatible function approximation**과 **gradient temporal-difference learning**을 소개한다.



#### On-Policy Deterministic Actor-Critic

일반적으로, deterministic policy를 따라 행동하면 **충분한 exploration을 보장할 수 없으며, local-optima로 유도되곤 한다**.

그럼에도 불구하고 SARSA actor-critic(on-policy actor-critic)을 첫번째 예로 든 이유는 교육적인 것(**쉬운 설명**을 위해)이 주된 목적이다.

하지만, **충분한 노이즈가 있는 환경**에서의 학습시킨다면 deterministic policy를 이용하더라도 충분히 탐험하는 효과가 있을 것이니 유용한 방법일 수 있다.



stochastic actor-critic에서와 마찬가지로 deterministic actor-critic에도 **actor와 critic** 두가지 요소를 가지고 있다.

critic은 action-value function을 추정하며, actor는 action-value function의 gradient를 최대화하는 방식이다.

특히, actor는 deterministic policy의 **파라미터를 stochastic gradient asent 방식으로 업데이트** 하게된다.

또한, action-value function은 미분 가능한 근사함수로 대체되며 critic은 이를 근사/추정 한다.
$$
\delta_t = r_t \space+\space \gamma Q^w(s_{t+1},a_{t+1}) \space-\space Q^w(s_t,a_t)
\\w_{t+1} =w_t\space +\space \alpha_w\delta_t\nabla_wQ^w(s_t,a_t)
\\\theta_{t+1} = \theta_t + \alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_aQ^w(s_t,a_t)|_{a=\mu_\theta(s)}
$$


#### Off-Policy Deterministic Actor-Critic





#### Compatible Function Approximation



