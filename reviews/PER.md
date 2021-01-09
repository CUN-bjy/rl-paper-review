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

이러한 방법을  **Prioritized Experience Replay(PER)** 라고 부르며 당시의 DQN보다 훨씬 좋은 성능을 자랑하는 SOTA로 자리매김 하였으며,

기존의 다른 model-free learning방식에 적용해 exploration 성능을 높이는데 매우 효과적으로 사용되고 있다.

</br>

### [Prioritized Replay]

#### (1) Prioritizing with TD-error

prioritized replay에서 가장 중요한 요소는 바로 **각 transition의 중요한 정도를 측정할 수 있는 평가지표**이다.

가장 이상적으로는 RL agent가 해당 transition으로 부터 얼마나 배울 수 있을 지에 대한 것이겠지만, 이는 쉽게 얻을 수 있는 지표는 아니다.

보다 현실적인 접근으로는 transition의 **magnitude of TD error**를 측정 하는 것이라고 할 수 있는데,

이는 해당 transition이 <u>얼마나 surprising 한지,  얼마나 예상 외의 것인지</u>를 의미한다.

</br>

TD error를 이용한 prioritized replay의 효과를 확인하기 위해 아래 그래프와 같이 *Blind Cliffwalk* 환경에서 

uniform과 oracle baseline, 그리고 '***greedy TD-error prioritization***' algorithm을 비교하였다.

</br>

이 알고리즘의 원리는 다음과 같다.

1. 매 transition을 따라 TD error를 계산해 replay memory에 저장한다. 

2. TD error의 크기가 가장 큰 transition은 memory로부터  replay된다.

3. 각 transition에는 q-learning update가 진행되며, 이는 TD error에 비례하도록 업데이트된다.

4. 알려진 TD error가 없는 new transition에 대해서는 maximal priority를 적용해 memory에 넣어준다.

   (모든 경험에 대해 적어도 1번 이상은 replay되도록 보장하기 위함이다)

<img src="../img/per1.png"/>

</br>

#### (2) Stochastic Prioritization

그러나 *greedy TD-error prioritization*에는 몇가지 문제점이 존재한다.

1. TD에러가 replay된 transition에 대해서만 업데이트 된다.(모든 memory에 대해 계산하기에 computation이 크기때문)
   - 이로인해 **처음에 TD-error가 낮게 평가된 transition에 대해 다시는 방문할 기회가 없게된다**!
2. **noise에 취약**하다.(rewards 가 stochastic한 경우!)
3. greedy prioritization은 experience memory의 극히 일부분에 집중하도록 만든다.
   - 이는 **다양한 경험을 충분히 전달하지 못하며 over-fitting에 빠질수** 있도록 만든다.

</br>

이어서 해당 논문에서는 이슈들를 극복하기 위해 greedy prioritization 방법과 uniform random sampling방법을 섞어

서로의 장단점을 보완한 ***stochastic sampling method***를 제안한다.

즉, replay memory 내의 transition의 priority는 유지하되, 

모든 transition에 대해서 non-zero 확률의 방문율을 가지도록 보장할 수 있도록 한다.

아래 수식은 k개의 transition 중 i번째의 transition의 sampling 확률을 나타내며, <img src="../img/alpha.png"/>는 얼마나 prioritization에 의한 sample을 많이 할 것인가를 결정한다.

<img src="../img/alpha.png"/>=0일 때, uniform case이며 ,<img src="../img/alpha.png"/>=1일 때, greedy prioritization이다.

<img src="../img/per4.png"/>

여기서 Pi 항은 두가지 방법으로 계산해 낼 수 있는데, 

직접적인 방법으로는 **proportional prioritization**방식으로 TD에러에 비례하지만 작은 constant값을 포함시켜줌으로써 모든 transition의 방문확률을 0이 아니도록 만들어주는 효과가 있다. 

<img src="../img/per5.png"/>

그 다음으로 간접적인 방법인 **rank-based prioritization**가 있다. 이는 TD에 따라 replay memory 내의 transition에 TD error에 따라 rank를 매기는 것이다.

<img src="../img/per6.png"/>

<img src="../img/per3.png"/>

</br>

#### (3) Annealing the Bias

**Prioritized Replay는 보통 편향치를 가져오는데**, 주로 expectation에 대한 distribution이 정형화되지 않은 상태로 update마다 바뀌기 때문이다.

해당 논문에서는 **importance-sampling(IS) weights**를 이용해 bias를 잡을 수 있었다.

<img src="../img/per7.png"/>

이는 (Mahmood et al., 2014)와 같이 *weighted* IS 방식으로 Q-learning에서 TD error 대신 weighted IS를 곱한 것을 이용해 업데이트하는데에 사용된다.

</br>

일반적인 RL 시나리오에서, <u>unbiased updates</u>는 training의 막바지에서 수렴하도록 하는데에 가장 중요한 역할은 한다.

해당 논문에서는 importance-sampling correction의 정도를 점진적으로 상승(*annealing*)시켜 training의 막바지에는 최대로 correction이 되도록 유도한다.

여기서 correction의 정도를 조절하는 Parameter는 <img src="../img/beta.png"/> 이며 선형적으로 상승해 training의 마지막에 1이 되도록 한다.

특히 prioritization에 대한 조절계수인 <img src="../img/alpha.png"/>와 함께 올려주면 더욱 확실하게 correcting이 된다고 한다.

</br>

neural network과 같은 비선형 근사함수와 prioritized replay를 함께 사용했을 때에 Importance Sampling의 **또 다른 이점**이 있다. 

gradient의 first-order approximation의 경우 일반적으로 local하게만 신뢰할 수 있기 때문에 large step으로의 학습은 성능이 좋지 않다.

그런데 prioritization의 과정에서 transition의 high-error가 learning step을 넓게 만들어주기도 하는데, 

이 과정에서 **IS correction이 gradient의 크기를 줄여줘 효과적으로 step의 크기를 줄여**준다는 것이다.

</br>

#### (4) Algorithm(PER)

<img src="../img/per2.png"/>



### [Experiments]

<img src="../img/per8.png"/>

### [Discussion]

*논문참고.*