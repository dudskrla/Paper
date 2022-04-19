# Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition

- 논문 : [2003.06606.pdf (arxiv.org)](https://arxiv.org/pdf/2003.06606.pdf)

# Abstract

```
(의의) we bridge the gap between the isolated processes of data augmentation and network optimization by joint learning

(원리) An agent network learns from the output of the recognition network and controls the fiducial points 
to generate more proper training samples for the recognition network
```

# Introduction

### Data augmentation의 중요성

data collection과 data annotation은 많은 시간과 비용을 필요로 한다. 특히, Object detection과 달리 text의 annotation 작업은 하나의 이미지에 많은 글자가 존재하므로, 상대적으로 더 어렵다. 이때, data collection이나 data annotation과 비교해서 data augmentation은 비교적 비용이 적게 드는 방법이 될 수 있다. 

### 기존 Data augmentation 방식의 문제점
<img width="475" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720716-d9ad4841-f55a-486d-a26a-b41af90b15ea.png">


1️⃣ Figure 1 (a)의 경우, 하나의 이미지에 있는 **많은 글자들**을 **하나의 집합**으로 생각하고 augmentation을 진행한다. text image에 대한 augmentation의 목표는 각 글자별 다양성을 높이는 것이므로, 이러한 방법은 적합하지 않다. 

2️⃣ long-tail distribution을 보이는 데이터에 random augmentation을 적용하면, manual하게 조절되는 static distribution을 이용해서 학습에 도움이 되지 않는 많은 easy sample을 만들어 낸다.

3️⃣ static distribution에 적합한 best augmentation을 찾았다고 하더라도, 또 다른 dataset에서는 예상한 대로 적용되지 않을 수도 있다. 


### 의의

- manual modification 없이 다른 dataset에도 자동으로 적응하는 learnable augmentation를 제안했다.  
- data augmentation과 모델을 jointly optimize하는 end-to-end trainable 프레임워크를 제안했다.
- 적은 학습 데이터로 학습한 모델의 성능을 향상시켰다.

# Methodology

### Overall Framework

<img width="890" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720762-1613557f-3332-4163-8188-90f074367cae.png">


세 가지 메인 module은 아래와 같다. 

- `agent network` → (더 어려운 학습 데이터를 생성하는) moving state distribution을 예측한다.
- `augmentation module` → moving state를 토대로 augmented image를 생성한다.
- `recognition network` → augmented image의 text를 예측한다.

1️⃣ 이미지에서 기준점(fiducial point)을 초기화한다. 

2️⃣ moving state(기준점의 이동 상태)를 `augmentation module`의 input으로 전달한다. 

1) moving state predicted by `the agent network` 

2) randomly generated moving state 

3️⃣ `augmentation module`이 image를 input으로 전달받고, 각 moving state를 토대로 transformation을 적용한 augmented image를 생성한다. 

4️⃣ `recognizer`가 augmented image의 text를 예측한다.

5️⃣ edit distance로 augmented image의 recognition difficulty를 측정한다. 

6️⃣ `agent`는 recognition difficulty를 높인 moving state로부터 학습하고, `recognizer`의 취약한 부분을 탐색한다. 이 과정에서 recognizer는 robust하게 된다.  

### Text Augmentation

<img width="489" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720809-d2f36ae1-575d-4903-b4de-a0c140f0c95d.png">


1️⃣ image를 N개의 patch로 averagely하게 나누고, 2(N+1)개의 기준점을 초기화한다. 이때, 기준점 p는 image 테두리 (각각 위와 아래)에 존재한다. 

2️⃣ predicted moving state distribution을 토대로 augmentation을 적용하고, 기준점 p를 (반지름 R 사이의 거리 내에서) 기준점 q로 랜덤하게 이동시킨다. 

(이동 방향) predicted moving state distribution에 기반해서 설정 

(이동 거리) 반지름 R 사이의 거리 내에서 랜덤하게 설정

3️⃣ augmented image를 생성하기 위해서, input image에 대해 moving least squares를 기반으로 한 similarity deformation을 적용한다. 이때, 연속함수(특히, 다항함수로)에 기반해서 생성하는 이유는 learnable 하도록 만들기 위해서이다. 

> **수식**
> 

image의 한 점 u에 대한 transformation은 아래와 같이 표현할 수 있다. 

<img width="380" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720836-f64c3c66-ec50-4558-b961-4467860a4020.png">


- M : linear transformation matrix
- p* : (이동하기 전의 기준점에 대해) **가중치가 적용된 기준점**
- q* : (이동한 후의 기준점에 대해) **가중치가 적용된 기준점**

<img width="454" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720854-c21b5749-992c-46d1-b272-6af327eaaaf6.png">


- w_i : 점 u에 대한 weight
    
    점 u가 p_i에 가까워질수록 weight w_i가 증가한다. 이를 통해, 점 u는 가장 가까운 기준점의 움직임에 의존한다는 것을 알 수 있다.
    
    (생각) ‘u가 p_i에 가까워질수록’이므로, p_i는 u의 가장 가까운 기준점이다. 또한, w_i가 증가한다는 의미는 의존도 (중요도)가 높아진다는 의미이다. 
    

<img width="373" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720862-37a9caf0-2a15-4cb8-9db7-3294fc07acbb.png">


- best augmentation는 아래의 식을 최소화하는 T(u)이다.

<img width="390" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720871-324bf480-afed-41cb-b03d-05f366a0beef.png">


> **Discussion**
> 

<img width="400" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720880-237d7259-f5fd-4f21-9a15-a69e7bfdf730.png">


Rigid transformation은 **상대적인 모양을 유지**하므로, 비교적 사실적으로 object를 표현하는 transformation이다. 이에 반해, Similarity transformation은 **각 글자마다 변형**을 적용할 수 있으므로, text image에 더 적합하다. 

### Learnable Agent

휴리스틱 알고리즘에서 영감을 받아, 가능한 모든 해결 방안 중에서 하나를 선택하는 방식으로 진행한다. 학습(training) 과정의 매 단계마다 predicted moving state에 변화를 준다. 이는 곧 학습 대상(learning target)의 후보군이 된다. 

random moving state가 recognition difficulty를 높이면, agent는 random moving state로 learning target을 설정하고 학습한다. 반대로, random moving state가 recognition difficulty를 낮추면 learning target을 역전시켜서, predicted moving state로 학습한다. 

각 기준점을 이동시키는 moving operation에는 두 가지 요소가 있다.

1) 이동 방향 (Delta{x}, Delta{y}) 

2) 이동 거리 (|Delta{x}|, |Delta{y}|) 

하지만, learning space에 이동 거리를 포함하지 않고, **이동 방향**으로 제한한다. 그 이유는,

1) 본 논문의 저자들이 실험한 결과, 이동 거리에 대한 학습은 수렴하지 않았기 때문이다.

2) 잘못된 agent의 경우, 항상 최대 이동 거리(maximum moving distance)를 예측해서 과도한 distorted sample을 생성했고, 이는 recognizer의 학습을 불안정하게 만들었기 때문이다. 

이동 거리는 반지름 R 범위 내에서, 이동 방향을 토대로 **랜덤**하게 지정된다. 이를 통해, agent가 단조로운 moving state distribution을 예측하는 것을 방지한다. 

> **learning scheme of the agent network**

<img width="400" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720884-f714fca5-aff6-43ba-b14a-ad6632a3a9c2.png">


1️⃣ agent가 더 어려운 학습 데이터를 생성하는 것을 목표로, moving state distribution을 예측한다. 

2️⃣ augmentation module은 random moving state와 predicted moving state를 input으로 받는다. 

3️⃣ augmentation module이 두 가지의 moving state를 토대로 augmented image를 각각 생성한다. 

4️⃣ recognizer가 augmented image를 input으로 받아서 text를 예측한다. 

5️⃣ (ground truth와 predicted text 사이의) edit distance로 recognition difficulty를 측정한다. 

6️⃣ agent가 (recognition difficulty를 높이는) moving state를 input으로 받아서, 가이드로 사용하고 업데이트된다. 

(추가) recognizer는 augmented image로 update 하고, agent는 edit distance가 높은 moving state로 update 한다. 

# Appendix

### long-tail distribution (class imbalance)

![x축은 class index (class에 속하는 데이터 수가 많은 순으로 정렬) / y축은 각 class에 속하는 데이터 수](https://blog.kakaocdn.net/dn/caJbJl/btrodlAMOvq/908F9jtwPkpDpU18kEObA0/img.png)

x축은 class index (class에 속하는 데이터 수가 많은 순으로 정렬) / y축은 각 class에 속하는 데이터 수

딥러닝 모델 학습을 위한 대다수의 데이터셋은 각 class에 속하는 데이터의 수가 고르게 분포되어 있다. 그러나 현실세계의 데이터는 그 분포가 고르지 않고, 일부 class에 속하는 데이터 수만 매우 많고 나머지 class에 속하는 데이터 수는 매우 적은 경우가 대부분이다. 이렇게 데이터 전체의 대부분을 차지하는 dominant한 class들을 **head class**라고 하고, 데이터 수가 매우 작은 class들을 **tail class**라고 한다. 

- 참고 : [[딥러닝 논문리뷰] Decoupling Representation and Classifier for Long-Tailed Recognition (ICLR 2020) (tistory.com)](https://bo-10000.tistory.com/109)

### edit distance

두 개의 문자열 A, B가 주어졌을 때 두 문자열이 얼마나 유사한 지를 알아낼 수 있는 알고리즘입니다. 이를 통해, 문자열 A가 문자열 B와 같아지기 위해서는 몇 번의 연산을 진행해야 하는 지 계산할 수 있습니다. ***여기서의 연산이란, 삽입(Insertion), 삽입(Deletion), 대체(Replacement)를 말합니다.*** 

기본적으로는 **두 데이터 사이의 유사도**를 알아내기 위해 사용할 수 있습니다.

- 참고 : [편집거리 알고리즘 Levenshtein Distance(Edit Distance Algorithm) (madplay.github.io)](https://madplay.github.io/post/levenshtein-distance-edit-distance)

### moving least squares

(원본) **Moving least squares** is a method of reconstructing [continuous functions](https://en.wikipedia.org/wiki/Continuous_function) from a [set](https://en.wikipedia.org/wiki/Set_(mathematics)) of unorganized point samples via the calculation of a [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares) [measure](https://en.wikipedia.org/wiki/Measure_(mathematics)) biased towards the region around the point at which the reconstructed value is requested.

(생각) sample로 주어진 점에 대해 weighted least square를 계산하여, 연속함수를 생성하는 것이다. 

![Untitled](https://user-images.githubusercontent.com/90603530/163720968-220c470f-b4e0-440f-8d0c-2b0c29fd46d7.png)  


<img width="463" alt="Untitled" src="https://user-images.githubusercontent.com/90603530/163720980-22054094-6917-452b-801f-22bf2d8ff108.png">  


점 (x_i, f_i) 과 다항함수의 차수가 m일 때의, moving least squares를 P(x)라고 하자. (단, P는 차수가 m인 모든 다항식 p에 대한 weighted least-square error를 최소화한다.) 이때, theta(s)는 weight이고, s→∞ 일 때, 0으로 수렴한다.

- 참고 : [Moving least squares - Wikipedia](https://en.wikipedia.org/wiki/Moving_least_squares)

### similarity transformation

변환시킨 후의 모양이 원래의 모양과 유사하게 되는 것이다. 

```
Similarity transformation = 회전(rotation) + 평행이동(translation) + 크기변화(scaling)
```

- 참고 : [다크 프로그래머 :: [영상 Geometry #3] 2D 변환 (Transformations) (tistory.com)](https://darkpgmr.tistory.com/79#:~:text=3.2%20Similarity%20Transformation%20(%EB%8B%AE%EC%9D%8C%20%EB%B3%80%ED%99%98,%EB%8F%84%ED%98%95%EC%9D%84%20%EC%83%9D%EA%B0%81%ED%95%98%EB%A9%B4%20%EB%90%A9%EB%8B%88%EB%8B%A4).)
