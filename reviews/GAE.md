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

1. 보통 아주 많은 수의 sample을 요구하며,
2. input data가 non-stationary하지 않는데 stable하고 steady한 improvement를 얻기가 너무 어렵다는 것이다.

</br>

