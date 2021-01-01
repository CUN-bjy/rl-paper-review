[HOME](../README.md)

### PER

[Prioritized Experience Replay]

Tom Schaul, John Quan, Ioannis Antonoglou and David Silver, Google DeepMind(2015)

[`PAPER`](https://arxiv.org/pdf/1511.05952.pdf) 

<br/>

### [Scheme]

기존의 DQN 논문에서는 Experience Replay를 사용하여 **과거의 경험을 기억**해 학습에 재사용할 수 있게 만들었다.

이는 agent의 exploration문제와 sample efficiency를 매우 크게 높일 수 있게 되었다는 의의가 있다.

</br>

기존에는 저장했던 경험의 중요성과 상관없이 random하게 추출하여 재사용해 학습에 사용해왔다.

본 논문의 **핵심**은 아주 간단하게 설명하자면, **보다 중요한 경험을 자주 재사용하도록 만들자!** 라고 할수 있겠다.

구체적으로는 현재의 Q Network를 이용해 기존의 경험과의 *TD-error*를 구한 후, 이를 사용해 prioritized buffer로 만들어 구현한다.

</br>

이러한 방법을  **Prioritized Experience Replay(PER) ** 라고 부르며 당시의 DQN보다 훨씬 좋은 성능을 자랑하는 SOTA로 자리매김 하였으며,

기존의 다른 model-free learning방식에 적용해 exploration 성능을 높이는데 매우 효과적으로 사용되고 있다.

</br>

### [Prioritized Replay]

#### 3.1 A Motivating Example



#### 3.2 Prioritizing with TD-error



#### 3.3 Stohastic Prioritization



#### 3.4 Annealing the Bias