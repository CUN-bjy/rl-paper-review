[HOME](../README.md)

### 5. TRPO

[Trust region policy optimization]

John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel (2015)

[paper_link](https://arxiv.org/pdf/1502.05477.pdf)

</br>

> 이번 리뷰부터는 너무 디테일하지 않으면서 전체적인 아이디어를 캐치할 수 있도록 써보려 한다.

### [Scheme]

이번 논문에서 다루는 trpo는 기본적으로 stochastic policy에 대한 policy optimization 기법이다.

Schulman이 제안한 trust region은 기본적으로 perfomance가 상승하는 방향으로의 update를 보장하는 optimization 기법에 대한 이야기이다. 이를 위해 그 전에는 어떤 문제점이 있었고 해결하기 위해 구체적으로 어떤 접근법을 사용하였는지가 이 논문을 보는 묘미가 될 것이다.

</br>

### [Preliminaries]

일반적으로 policy의 performance의 척도를 평가하기위해 reward를 반영해 performance function을 만드는데,

stochastic policy에 대한 expected discounted reward의 표현은 다음과 같다.

<img src="../img/trpo1.png"/>

또한, policy <img src="../img/pi.png"/>에 대해 advantage를 취한 policy를 <img src="../img/adv_pi.png"/>라 할 때, 시간에 따라 축적된 expected return은 다음과 같다.

<img src="../img/trpo2.png"/>

이 수식의 의미는 모든 state에 대해 nonnegative expected advantage를 가진다면 policy performance의 상승을 보장할 수 있다는데에 있다. 물론.. 추정과 근사를 하는 과정에서 error가 존재하므로, 몇몇 state에 대해 negative expected advantage를 가지는것은 불가피하다.

</br>

아래 식은 performance function <img src="../img/eta.png"/>에 대한 local approximation 식이다. advantage policy <img src="../img/adv_pi.png"/>로부터 직접 sample을 얻어오는것이 어려우므로, 간접적으로 local policy <img src="../img/pi.png"/>로부터 얻으려는 것이다.

<img src="../img/trpo3.png"/>

또한, policy가 parameterized policy이며 미분가능하다면 performance <img src="../img/eta.png"/>의 1차 미분식과 일치한다.

<img src="../img/trpo4.png"/>

위 식은 충분히 작은 step으로 approximation이 증가하는 방향으로 움직이면, performance <img src="../img/eta.png"/>역시 증가한다는 것을 나타낸다. 하지만 얼만큼의 big step까지 이를 보장하는지에 대한 guidance가 없다는 한계가 있다.

</br>

이 문제를 해결하기 위해  Kakade&Langford(2002)가 **conservative policy interation** 이라는 개념을 제시했다.

바로 위에서 보장하지 못했던 내용에 대한 lower bound를 나타낼 수 있는 식을 제시 하였는데, 기존의 policy를 업데이트함에 있어 approximation error에 의해 Improvement가 보장되지 못할 수 있는 상황을 미연에 방지할 수 있다.

<img src="../img/trpo5.png"/>

이 식을 본 Schulman은 mixture policy를 업데이트하는데 뿐만 아니라 모든 general stochastic policy에 적용할 수 있는 가능성을 보았고 이를 사용하기로 하였다.

</br>

### [Monotonic Improvement Guarantee]

이번에는 위 식 *conservative policy iteration bound*를 어떻게 general stochastic policy에 적용되는 지  알아보자.

 기존 식에서의 <img src="../img/alpha.png"/>는 기존 policy와 새로 current policy간의 업데이트 비율을 결정하는 parameter로서 사용되었다.

이번에는 <img src="../img/alpha.png"/>를 <img src="../img/pi.png"/>와 <img src="../img/adv_pi.png"/>간의 distance measure로, <img src="../img/epsilon.png"/>을 적절히 변형해주어 아래와 같은 식을 도출하였다.

그리고 두 policy간 distance measure는 **total variation divergence**를 이용하였다.(증명은 appendix 참고)

***Theorem 1.***

<img src="../img/trpo6.png"/>

다음으로 total variation divergence 와 **KL divergence** 사이에는 아래와 같은 관계식이 성립한다.

<img src="../img/trpo7.png"/>

이를 응용해 theorem 1에 대입하면 다음과 같은 식이 성립된다.

<img src="../img/trpo8.png"/>

이 식을 이용해 policy iteration을 진행하는 알고리즘 식은 다음과 같으며, 이 식을 통해 지속적으로 performance 증가를 보장하는 policy iteration을 보일 수 있게 되었다.

<img src="../img/trpo9.png"/>

즉, <img src="../img/trpo10.png"/>일 때, 매 iteration마다 M을 최대화 함으로써 true objective <img src="../img/eta.png"/>가 non-decreasing임을 보장할 수 있게 되며, 이는 일종의 minorization-maximization(MM)알고리즘임을 알 수 있다.

MM algorithm에서의 용어로는 함수 M이 바로 surrogate function이라 불린다.



### [Optimization of Parameterized Policies]


$$
M_i(\pi) =  L_{\pi_i}(\pi) -CD^{max}_{KL}(\pi_i,\pi)
$$
