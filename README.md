# pg-paper-review

### Paper Review List

reference link: [PG Travel Guide](https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/)

- [x] [1. Sutton_PG](#1.-Sutton_PG)
- [ ] [2. DPG](#2.-DPG)
- [ ] [3. DDPG](#3.-DDPG)
- [ ] [4. NPG](#4.-NPG)
- [ ] [5. TRPO](#5.-TRPO)
- [ ] [6. GAE](#6.-GAE)
- [ ] [7. PPO](#7.-PPO)



> paper review를 하며 가능한 key idea라고 생각하는 부분들만 추렸으니 논문을 참조해가며 읽어주세요



### 1. Sutton_PG

[Policy gradient methods for reinforcement learning with function approximation]

Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour,1994

[paper link](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)



서문)

이 논문이 발표될 당시 RL의 주된 접근 방식은 '*value-function* approach'였다.

하지만, 이러한 방식에는 몇가지 **한계점**이 존재했는데,

**첫번째**는 optimal policy는 주로 확률적(stochastic)으로 몇가지 액션을 고르는 것인 경우가 많은데, deterministic policy를 찾는것에만 편향되어있다는 것이다.

**두번째**는 임의의 어떤 액션에 대한 보상기대값이 아주 근소한 차이만 나더라도, 해당 액션이 선택되고 안되고를 결정할 수 있다는 것이다. 이러한 불연속적인 차이는 'value-function approach'를 따르는 알고리즘의 수렴성을 입증하는데에 주요 문제점으로 꼽히고 있다. (**policy의 variance가 크다**)

이 논문은 이러한 문제점을 해결할 수 있는 대안적인 방법을 소개한다.

기존의 방식은 value function을 근사해 policy를 결정하는 반면, 독립적인 function approximator를 이용해 stochastic policy를 직접 근사한다.

가령, **input**: representation of states / **output**: action selection probabilities / **weights**: policy parameters

와 같은 형태의 neural network으로 표현하며,

 *policy gradient* approach를 이용해 다음과 같이 policy parameter를 퍼포먼스 방향으로의 gradient에 비례하도록 parameter를 업데이트 해준다면

<img align="center" src="./img/latex1.png">

performance(reward)가 극대화 하는 지점에서 local optimal policy로 수렴하게 될 것이다.

이는 *value-function* approach와 다르게 **parameter의 작은 변화가 policy의 작은 변화로 연결된다!**

(value-based에서는  parameter가 value를 이루므로 value가 조금만 변해도 policy의 차이가 많이 나는 문제가 있었음) 



본론)

본론에서는 새로운 내용들 보다는 서문에서 다루었던 내용들의 수학적 증명으로 이어진다.

1. Policy Gradient Theorem

   value function approximation을 하기위한 방법에는 두가지 방법이 있는데, **average-reward formulation**과 **start-state formulation** 두가지 방법이 있다.

   **Theorem 1 (Policy Gradient)**

   <img align="center" src="./img/latex3.png">
   
   이 수식은 결국 위 두가지 approximation 방법 모두에 적용이 가능하며 appendix에 두가지 방법을 이용해 모두 증명해놓았다. 
   
   <img align="center" src="./img/latex2.png">

   기본적으로 두 증명 모두 위 식과 같이 value function과 action-value function의 기본 정의로부터 유도된다. 유도하는 방식은 Q-function을 어떤 방정식으로 구해내느냐에 따라 달라지니 appendix를 유심히 살펴보시길 바란다. 결국 Theorem 1이 의미하는 바는 논문에도 잘 설명되어있듯이, **distribution of states가 policy changes에 어떤 영향도 주지 않는다는 것이다.** 이는 **sampling을 통해 gradient를 근사하기에 편리한 특성**을 지닌다는 뜻이다. 물론 Q-function도 반드시 추정되어야 하는데 이를위한 한가지 방법은 실제 reward값을 이용하는 것이다.(???)
   
    
   
2. Policy Gradient with Approximation

   

3. Application to Deriving Algorithm and Advantages

   

4. Convergence of Policy Iteration with Function Approximation



### 2. DPG

[Deterministic policy gradient algorithms]



### 3. DDPG

[Continuous control with deep reinforcement learning]



### 4. NPG

[A natural policy gradient]



### 5. TRPO

[Trust region policy optimization]



### 6. GAE

[High-Dimensional Continuous Control Using Generalized Advantage Estimation]



### 7. PPO

[Proximal policy optimization algorithms]