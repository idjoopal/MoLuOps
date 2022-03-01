# Week 1.  Fundamentals

## Lecture 1 : DL Fundamentals

### Neural Networks 신경망

- 단일 뉴런(생물학적)
  - 뉴럴 네트워크는 뉴런과 비슷하게 생겨서 그렇다.
  - Dendrites를 통해 정보를 받아서 전기신호를 axon이라 불리는 다리를 통해 전달
  - 터미널을 통해 다른 뉴런에게 신호를 전달함
- 인공 뉴런
  - 이 뉴런의 움직임은 수학적으로 간단한 함수( 퍼셉트론)으로 표현할 수 있음
  - weight를 줌으로 써 dendrrte의 역할을 함
  - b는 bias
  - 활성화 함수는 일종의 threshold 함수로써, 신호를 전달할지 전달하지 않을지 작동함
- 일반적인 활성화 함수
  - 시그모이드, 매우 간단, 아웃풋은 0~1로 변경됨
  - 하이퍼볼릭 탄젠트도 비슷하게 많이쓰임
  - 최근에는 ReLU를 많이 사용함.
    - max 함수로서, 0보다 작을땐 0, 크면 max
- 뉴럴 네트워크
  - 모든 퍼셉트론이 각각의 가중치와 vias를 가지고 있고, 이것들이 인풋을 바꿔줌.
  - 함수는 y로 대표됨

### Universality 보편성

- 신경망은 함수 y = f(x,w)로 대표됨
- 하지만 그지같이 위아래로 오르내리는 그래프는 어떻게 할수 있을까?
-  Universal Function Approximation Theorem
  - 어떠한 연속 함수 f(x)가 주어졌을 때, 2개의 층 신경망이 충분한 은닉노드를 보유하고 있다면,  f(x)에 근사하게 할 수 있는 weight를 가지고 있다.
  - 이러한 이론이 옳다는 것은 수박도에서도 증명하고 있다.(http://neuralnetworksanddeeplearning.com/chap4.html)

### Learning Problems 학습 문제

- 지도, 비지도, 강화 학습 큰 세가지 카테고리
  - 지도학습은 상용되었음
  - 강화학습은 나중이야기
- 지도학습 예시 - 스킵
- Unsupervised Learning X
  - 예시1 RNN을 통해 다음 글자를 예측하는 모델
  - 예시2 단어를 벡터화 하여 단어간 관계를 이해하고 비슷한 단어를 예측하는 것.
  - 예시3 다음 픽셀을 예측하기
  - 예시4 오토인코더 Latent vector(잠재요인)
  - 예시5 GAN 모델
- 강화학습 예시 - 스킵

### Empirical Risk Minimization / Loss Functions  경험에 의거한 위험 최소화/손실함수

- 선형회귀에서 "그 선"을 정하는 이유
  - 데이터와 가장 잘맞고, squared error 가 가장 최소화됨
- 신경망 회귀
  - 손실을 최적화하기 위한 가중치와 편향 찾기
    - 일반적으로 MSE나 Huber loss를 손실함수로 잡는다
- 신경망 분류
  - 손실을 최적화하기 위한 가중치와 편향 찾기
    - 범주형 output이기 때문에 크로스-엔트로피를 손실함수로 잡는다.

### Gradient Descent

- 우리는 w와 b를 찾아 손실을 최적화 해야함
- 40p. 현재 w에 alpha(학습률)을 적용한 weight에 대한 손실함수의 기울기를 빼주어 w를 업데이트
  - 이걸 통해 점점 손실을 최소화 할 수 있다.
- Conditioning
  - 데이터를 동일한 분산, 평균 0으로 Conditioning해야 학습하기가 좋음
  - Normalization
    - Batch norm
    - Weight norm
    - Layer norm
  - 두번째 방법
    - 뉴턴의 방법, 자연경사 -> 이방법들은 너무 계산이 복잡
    - Adagrad, Adam, Momentum의 방식을 사용함.
- 경사하강의 샘플링 계획
  - 일반적인 경사 하강법은 모든 스텝에 계산이 복잡해짐
  - 확률적 경사하강법
    - 각 step마다 전체 데이터 대신 배치 데이터를 사용하여
    - 계산량을 줄이고 스텝의 노이즈를 증가시킴
    - 결과적으로 계산량 비례 진행속도를 개선할 수 있음.

### Backpropagation / Automatic Differentiation  역전파, 자동 미분

- 확률적 경사하강법을 사용하여 손실함수를 최적화해서 학습을 줄임.
- 경사를 효과적으로 계산하는 방법은? -> 자동미분
- 딥러닝 모델은 최종 output이 간단한 연산의 sequential한 chain으로 표현된다.
  - Chain Rule을 통해서 해결
  - 우리는 이걸 직접 구현할 필요가 없다
    - 파이토치, 텐서플로 등이 지원해줌
  - Backpropagation : 함수의 순방향으로 계산을 하면서 캐싱한 다음 이것을 역방향으로 되돌아가면서 진행

### Architectural Considerations (deep / conv / rnn)  구조적인 고려사항

- 가장 간단한 모델
  - 풀 연결 층들의 모임
  - 극단적으로 큰 네트워크는 모든걸 만들수있지만, 그럴 수 없음
- 랜드스케잎 최적화 / 컨디셔닝
  - 넓이, 연결 스킵, Normalization
- 계산적, 파라미터
  - Factorized Conv
  - Strided Conv

### CUDA GPU를 사용하기 위한 알고리즘

- 왜 2013년에 딥러닝의 폭발이 시작했는가?
  - 더 큰 데이터
  - GPU에서의 행렬계산을 위한 좋은 라이브러리
- GPU가 중요한 이유?



## LAB 1 Introduction

- 문제를 이해하고 환경을 세팅하고 코드를 보고 mnist 학습하기
- 손글씨 종이를 인식하여 텍스트화 하기
  - 1. 웹에서 POST를 request로받아 이미지를 디코드
    2. 모델이 라인을 찾고 텍스트화
    3. response로 encode하여 클라이언트에 답변