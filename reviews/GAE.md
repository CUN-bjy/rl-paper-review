[HOME](../README.md)

### 6. GAE

[High-Dimensional Continuous Control Using Generalized Advantage Estimation]

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel(2016)

[paper_link](https://arxiv.org/pdf/1506.02438.pdf)

</br>

- 참고할만한 링크:
  - https://m.blog.naver.com/PostView.nhn?blogId=kwonpub&logNo=221298006981&proxyReferer=https:%2F%2Fwww.google.com%2F





### [Abstact]

policy gradient method은 RL에서 아주 매력적인 접근법이다.

그 이유는

1. neural network를 nonlinear function estimator로 바로 사용가능하며,

2. 누적 reward를 직접적으로 최적화 할 수 있도록 만들어준다.

</br>

하지만, 두가지 아주 중요한 challenges가 남아있는데,

1. 보통 아주 많은 수의 sample을 요구하며,(sample complexity가 높기 때문)
2. input data가 non-stationary하지 않는데 stable하고 steady한 improvement를 얻기가 너무 어렵다는 것이다.

</br>

이 논문에서는 첫번째 문제를 다루기 위해 policy gradient estimation의 variance를 줄여주는 value function을 사용한다.(약간의 bias를 가지게된다). 이 value funtion은 TD lambda와 유사한 advantage funtion의 exponentially-weighted estimator를 사용한다.

다음으로 두번째 문제를 해결하기 위해 policy와 value function의 업데이트를 위해 trust region optimization을 이용한다.

</br>

이러한 접근법을 통해 highly challenging 3D locomotion tasks에 대해 아주 좋은 결과들을 도출해 내었다.(learning gaits for bipedal, quadrupedal, learning standing up motion for biped)

지난 결과물들은 직접 policy가 나타내는 것을 직접 연결해줘야 했던것에 비해 이들은 fully-model-free 하다.

</br>

### [Introduction]

reinforcement learning에서의 목적은 policy의 expected total reward를 maximize하는 것이다.

위 문제를 푸는데 있어 가장 어려운 요소가 있다면, 어떤 action을 행했을 때 그것이 reward에 미치는 영향에 대해 long time delay가 존재한다는 것이다. 이것을 **credit assignment problem** 또는 distal reward problem이라 부른다.

value function은 이 문제에 대해 아주 멋진 solution을 내놓는데,  바로 delayed reward가 도달하기 전에 action의 좋은 정도를 추정하는 것이다.

</br>

parametized stochastic policy를 사용할 때, expected total returns의 gradient를 unbiased하게 추정하는것이 가능하다.(williams,1992; sutton et al.,1999)

유감스럽게도, 이는 시간에 따라 variance가 의도치않게 커지는 문제를 가지고 있었다. 과거의 action과 미래의 action이 미치는 영향이 혼재되어있기 때문이다.

actor-critic method라 불리는 방법은 empirical returns가 아닌 value funtion을 사용함으로써 약간의 bias를 가지지만 varience를 줄이는 데에 성공했다. (Konda&Tsitsiklis, 2003; Hafner&Riedmiller,2011)

하지만, sample의 수가 많아짐에 따라 high variance는 피할수 없는 문제였고, bias는 안좋은 영향을 미쳤다..(이 bias는 수렴하지 못하게하거나, 안좋은 방향으로 수렴하게 만들었다..)

</br>

이 논문에서는 또 하나의 policy gradient estimator를 제시하려고 하며, 이는  variance를 확실히 줄여주고, 용인 가능한 수준의 bias를 가지게 된다.