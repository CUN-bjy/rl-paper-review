[HOME](../README.md)

### 5. TRPO

[Trust region policy optimization]

John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel (2015)

[paper_link](https://arxiv.org/pdf/1502.05477.pdf)

</br>

> 이번 리뷰부터는 너무 디테일하지 않으며 전체적인 아이디어만 캐치할 수 있도록 써보려 한다.

### [Scheme]

이번 논문에서 다루는 trpo는 기본적으로 stochastic policy에 대한 policy optimization 기법이다.

일반적으로 policy의 performance의 척도를 평가하기위해 reward를 반영해 performance function을 만드는데,

stochastic policy에 대한 expected discounted reward의 표현은 다음과 같다.
$$
\eta(\pi) =  \mathbb{E}_{s_0,a_0,...}[\sum^\infin_{t=0}\gamma^tr(s_t)], \text{where}
\\ s_0 \sim \rho_0(s_0), a_t \sim \pi(a_t|s_t), s_{t+1} \sim P(s_{t+1}|s_t,a_t)
$$
또한, policy에 대해 advantage를 취한 policy를 <img src="../img/npg1.png"/>
$$
\eta(\overset{\sim}\pi) = \eta(\pi) + \mathbb{E}_{s_0,a_0,...\sim\overset{\sim}\pi}[\sum^\infin_{t=0}\gamma^tA_\pi(s_t,a_t)]
$$
