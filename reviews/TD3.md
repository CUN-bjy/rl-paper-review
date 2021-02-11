[HOME](../README.md)

### TD3

[Addressing Function Approximation Error in Actor-Critic Methods]

Scott Fujimoto , Herke van Hoof , David Meger (2018)

[`PAPER`](https://arxiv.org/pdf/1802.09477.pdf)   |	[`CODE`](https://github.com/CUN-bjy/gym-td3-keras)

<br/>

### [Scheme]

- John Schuleman의  TRPO, PPO 이후 policy gradient에서 가장 주목받고 있는 알고리즘들 중 하나인 TD3.
- 기존 DeepMind에서 발표한 DDPG와 거의 같은 형태를 가지고 있기 때문에 이론적 배경이나 구현과정이 쉬운 편.
- DDPG와의 차별점으로는 크게 두가지이다,
  - **Clipped Double Q-network and Update**
  - **Delayed Policy update**
  - 그리고 Target Policy Smoothing
- 그러니 부디 [DDPG](./DDPG.md)를 먼저 공부하고 오는게 좋을 듯 하다.

</br>

### [Abstact]

- DQN과 같은 value-based 방식의 강화학습 방법에서는 function approximation의 오차가 **value가 overestimated되도록 추정**하도록 만드는 주된 이유이며, 이는 suboptimal policy로 빠지도록 만든다.

- 이러한 현상은 actor-critic 방식에서도 마찬가지임을 이 논문에서 보이며, value-based 방식에서의 위와 같은 문제를 효과적으로 해결한 Double Q-learning의 이론을 기반으로 새로운 메커니즘을 제안한다.

  :arrow_right:*Clipped Double Q-network*

- 또한 이 논문에서는 target network와 overestimation bias의 연관성을 보이고 policy update를 지연시킴으로서 update 당 에러를 줄여 성능을 보다 향상시킨다.

  :arrow_right:Delayed Policy update

- 위 알고리즘은 OpenAI gym task를 통해 검증하였으며 기존의 SOTA보다 좋은 성능을 나타낸다.

</br>

### [Background]

기본적인 강화학습의 환경은 아래와 같이 표현된다.
$$
t : \text{at discrete time step}
\\s \in S : \text{with given state}
\\a \in A: \text{the agent selects actions}
\\ \pi : S \to A \text{, with respect to its policy}
\\r : \text{receiving a reward, } s' : \text{and the new state}
$$

$$
R_t = \sum^T_{i=t}\gamma^{i-t}r(s_i,a_i) : \text{discounted sum of rewards(return)}
$$

강화학습에서의 **목표**는 반환값의 기대치를 최대화 해줄 수 있는 최적의 policy $\pi_\phi$(즉 이를 구성하는 파라미터 $\phi$)를 구하는 것이며, 이러한 parameterized policy는 **반환값의 기대치의 기울기값을 통해 업데이트** 해준다.
$$
\text{expected return : } J(\phi) = \mathbb{E}_{s_i\sim p_\pi,a_i\sim\pi}[R_0]
\\\text{its gradient : } \nabla_\phi J(\phi) = \mathbb{E}_{s\sim p_\pi}[\nabla_aQ^\pi(s,a)|_{a=\pi(s)}\nabla_\phi\pi_\phi(s)].
$$
그리고 반환값의 기대치는 일반적으로 critic 또는 value function이라 불리는 Q-value를 계산해 구한다.
$$
Q^\pi(s,a) = \mathbb{E}_{s_i\sim p_\pi,a_i\sim\pi}[R_t|s,a] :\text{as definition}
\\ Q^\pi(s,a) = r + \gamma\mathbb{E}_{s',a'}[Q^\pi(s',a')], \space a'\sim\pi(s') : \text{bellman form}
$$
large state space에 대해 Q value를 근사하기 위해 parameter $\theta$를 사용하며, objective $y$에 대해 업데이트 수식은 다음과 같다.
$$
y = r + \gamma Q_{\theta'}(s',a'), \space a'\sim\pi_{\phi'}(s')
$$
여기서 $\theta'$와 $\phi'$는 각각 target network에 해당되며 업데이트의 안정성을 위해 존재하며 현재의 actor, critic으로부터 각각 soft update받게된다(DDPG참고)
$$
\text{soft update : } \theta' \gets \tau\theta + (1-\tau)\theta'
$$

</br>

### [Overestimation Bias]

discrete action을 사용하는 Q-learning에서는 가치추정단계에서 다음과 같은 greedy target을 사용한다.
$$
y = r + \gamma max_{a'}Q(s',a')
$$
하지만, 이러한 방식은 target이 error $\epsilon$에 취약한 경우 error를 가지는 value의 maximum값이 일반적으로 true maximum보다 크다는 것이 입증되었다. *(Thrun & Schwatz, 1993)*
$$
\mathbb{E}_\epsilon[max_{a'}(Q(s',a')+\epsilon)] \ge max_{a'}Q(s',a')
$$
즉, 이러한 결과는 **지속적인 overestimation bias**를 만들게 되며 이러한 문제는 function approximation을 이용하게 되었을 때 피할 수 없는 문제이다.

</br>

위와 같은 문제가 자명히 밝혀진 반면에, **actor-critic방식에서도 overestimation bias 효과가 발생하는지**에 대한 연구가 없었으며 이 논문에서 밝히고자 한다는 것이 해당 section의 요지이다.

</br>

#### Overestimation Bias in Actor-Critic

먼저, 해당 section에서는 deterministic policy gradient를 사용해 policy를 업데이트한다는 가정 하에 overestimation이 발생한다는 것을 증명하려한다는 것을 알린다.

주어진 policy parameter $\phi$에 대해 function approximation을 사용해 update를 진행한 policy $\phi_{\text{approx}}$와 optimal policy $\pi$의 true value function을 사용해 update를 진행한 $\phi__{\text{true}}$가 아래와 같이 존재한다고 가정하자.
$$
\phi_{\text{approx}} = \phi + \frac{\alpha}{Z_1}\mathbb{E}_{s\sim p_\pi}[\nabla_\phi\pi_\phi(s)\nabla_aQ_\theta(s,a)|_{a=\pi_\phi(s)}]
\\\phi_{\text{true}} = \phi + \frac{\alpha}{Z_2}\mathbb{E}_{s\sim p_\pi}[\nabla_\phi\pi_\phi(s)\nabla_aQ^\pi(s,a)|_{a=\pi_\phi(s)}].
$$
그리고 위 parameter를 사용하고있는 policy를 각각 $\pi_{\text{approx}}, \pi_{\text{true}}$라 할 때 다음이 성립한다.

1. 충분히 작은 error $\epsilon_1$이 존재($\alpha\ge\epsilon_1$)한다고 할 때, $\pi_\text{approx}$의 approximate value는 $\pi_\text{true}$의 approximate value보다 작거나 같다.
   $$
   \mathbb{E}[Q_\theta(s,\pi_{\text{approx}}(s))] \ge \mathbb{E}[Q_\theta(s,\pi_{\text{true}}(s))]
   $$
   

2. 마찬가지로 충분히 작은 error $\epsilon_2$이 존재($\alpha\ge\epsilon_2$)한다고 할 때, $\pi_\text{true}$의 true value는 $\pi_\text{approx}$의 true value보다 작거나 같다.
   $$
   \mathbb{E}[Q^\pi(s,\pi_{\text{true}}(s))] \ge \mathbb{E}[Q^\pi(s,\pi_{\text{approx}}(s))]
   $$

3. 마지막으로 $\mathbb{E}[Q_\theta(s,\pi_{\text{true}}(s))] \ge \mathbb{E}[Q^\pi(s,\pi_{\text{true}}(s))]$이 성립한다면, 위 두식에 의해 $\alpha < min(\epsilon_1,\epsilon_2)일 때 value estimation 값은 overestimated될 것이다.
   $$
   \mathbb{E}[Q_\theta(s,\pi_{\text{approx}}(s))] \ge \mathbb{E}[Q^\pi(s,\pi_{\text{approx}}(s))]
   $$

</br>

수식으로 어렵게 씌여져있지만, 결국 전달하고자하는 바는 **function approximation을 이용한 actor-critic역시 true Q-value에 비해 overestimation됨**을 증명하고자 한 것이다.

</br>

 

![](../img/td3_1.png)



</br>

### [Addressing Variance]

