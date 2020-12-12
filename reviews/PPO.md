[HOME](../README.md)

### PPO

[Proximal policy optimization algorithms]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

[paper_link](https://arxiv.org/pdf/1707.06347.pdf) | [implementation](https://github.com/CUN-bjy/gym-ppo-keras)

<br/>

### [Scheme]

TRPO에 이은 John Schulman의 새로운 Policy Optimization 알고리즘이며, Schulman은 꾸준히 자신이 세운 SOTA 기록들을 갱신하며 넘사벽의 존재가 되어버렸다.

PPO가 나온지 3년이 지난 2020년 현재. TD3, SAC 등 좋은 알고리즘들이 많이 등장했지만, 쉬운 구현과 상징성의 이유로 아직까지 제일 유명한 알고리즘으로서 꼽힌다.

</br>

기존의 [TRPO](./TRPO.md)는 기본적으로 복잡하다.(구현도 어렵고 이론적으로 쉽지 않다)

또한 noise를 포함하는 구조(dropout 등)와 호환되지 않고, 다른 알고리즘들과 파라미터를 공유하지 않아 일반적이지 않다.

해당 논문에서는 PPO(Proximal Policy Optimization)라는 알고리즘을 제안하는데,

first-order optimization만을 사용해 구현이 매우 쉬우며 **TRPO만큼의 성능**을 지니면서 **data efficiency문제와 위의 문제점들을 해결**하였다.

</br>

### [Background: Trust Region Methods]

TRPO 논문에서는 아래의 식과 같이 특정 constraint 이내에서 objective funtion("surrogate" objective)가 최대화 되는 policy로 update를 진행하며,

이를 **Trust Region Methods**라고 한다.

<img src="../img/ppo1.png"/>

여기서 <img src="../img/theta_old.png"/>가 update이전의 policy parameter vector인데, 

해당 문제는 <u>conjugate gradient algorithm</u>을 사용해 근사적으로(objective는 linear approximation, constraint는 quadratic approximation) 해결한다. 

(즉, 계산이 많고, 엄청 복잡하다. 이는 근사적으로 문제가 해결되었지만, 다른 연구자들이 개선하거나 분석하기에 최악일 수 있다..)

</br>

사실 Schulman은 처음 TRPO를 정의내릴 때 <u>특정 surrogate objective가 policy <img src="../img/pi.png"/>의 성능을 보장하기 위한 lower bound를 형성한다</u>는 개념을 따라

constraint방법이 아니라 아래와 같이 **penalty 방법**(coefficient<img src="../img/beta.png"/>를 지님)을 제시하였다.

<img src="../img/ppo2.png"/>

그럼에도 constraint 방법을 사용한 이유는 task에 따라 policy의 scale이 달라질 수 있어 <img src="../img/beta.png"/>를 결정하기 어렵기 때문이다.

</br>

따라서 이 논문의 목적인 **`TRPO와 같이 성능보장이 가능한 policy optimization을 first-order algoritm로 계산하자`**를 만족하려면,

고정된 coefficient <img src="../img/beta.png"/>를 선택하는 문제와 SGD를 이용한 penalized objective의 최적화가 가능해져야만 한다.

<u>추가적인 개선이 필요하다!</u>

</br>

### [Clipped Surrogate Objective]

probability ratio를 다음과 같이 정의해보자.

<img src="../img/ppo3.png"/>

그리고 이를  기존의 "surrogate" objective함수에 적용하면 다음과 같이 표현해 볼 수 있을 것이다.

<img src="../img/ppo4.png"/>

여기서 **CPI**는 conservative policy iteration의 약자이며 TRPO에서 언급된 바 있다.

constraint 없이 위 objective를 maximization하게되면 policy update이 큰 스텝으로 진행될 가능성이 있다. 이는 monotonuous improvement를 보장할 수 없다.

</br>

그럼 objective를 어떻게 수정해야할까? 바로, probability ratio를 숫자 1에서부터 멀리 떨어져있는(**기존 policy와 많이 다른) policy에 penalty를 주는 것**이다.

아래는 이 논문에서 제시한 새로운 surrogate objective funtion이다. 

<img src="../img/ppo5.png"/>

여기서 <img src="../img/epsilon.png"/>은 hyperparameter이다.(default:0.2)

위 식에서 첫번째 term은 기존의 objective이며, 두번째 term은 clipped probability ratio를 적용한 것이다.

결과적으로 이 둘을 비교해 더 작은값을 취함으로써 **lower bound**를 형성하는 데에 의미가 있다.

아래 도표를 참고한다면 직관적으로 이해에 도움을 줄 수 있을 것이다.

<img src="../img/ppo6.png"/>



<img src="../img/ppo7.png"/>

</br>

### [Adaptive KL Penalty Coefficient]



</br>

### [Proximal Policy Optimization]

</br>