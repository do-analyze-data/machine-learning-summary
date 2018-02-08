# 머신러닝의 주요 개념
- 모델, 손실함수, 최적화, 모델평가

## 모델
- 데이터를 어떻게 바라볼지에 대한 가정
	- 문제를 바라보는 관점
	- 데이터에 대한 가정을 한데 모은 것
	- 머신러닝의 과정에서 정의 하는 것
		- 머신러닝의 과정
			- 모델 정하기 → 모델 수식화 하기 → 모델 학습하기 → 모델 평가하기
				- <text style="color:lightgray;">위의 과정을 반복함</text>
	- 데이터에 대한 가정의 집합
	- 머신러닝의 러닝은 <text style="color:red;">학습</text>은 데이터를 통해 추측하는 것
		- 학습은 모델 중에서 가장 데이터에 적합한 모델을 고르는 과정
		- 모델 = 함수 <text style="color:lightgray;">(similar)</text>

### 모델의 종류
- 간단한 모델
	- 데이터 구조가 간단
	- ex)linear model<text style="color:lightgray;">(선형 모델)</text>
		- 대표적으로 linear regression이 있음
	- 장점
		- 이해가 쉬움
		- 모델 자체의 변화폭이 적음, 데이터의 변화에 비해
			- 예외 데이터가 들어와도 영향을 적게 받음
		- 학습이 쉬움 
	- 단점
		- 가정 자체가 강력함
			- 모델의 표현 능력에 제약이 많음
- 복잡한 모델
	- 모델의 유연성을 중시
	- ex) [decision tree<text style="color:lightgray;">(결정 트리)</text>](https://ko.wikipedia.org/wiki/결정_트리_학습법)
	- 장점
		- 복잡한 데이터 모델링에 적합
	- 단점
		- 데이터의 모든 부분에 대해 일일이 모든 가정을 만듦으로써 노이즈까지 학습하여 > 성능 저하의 요인이 됨
- 구조가 있는 모델
	- 데이터 구조 자체를 모델링하는 모델
	- 입력과 출력 요소가 서로 연관관계가 있는 것
	- ex) Sequence model, Graphical model

#### 구조가 있는 모델
- [Sequence model<text style="color:lightgray;">(S2S model)</text>](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/seq2seq/)
	- 상태에 따라 출력값이 결정됨
		- 현재의 상태는 바로 직전의 상태와 현재의 입력 데이터에 영향을 받음
- [Graphical model](http://norman3.github.io/prml/docs/chapter08/0.html)
	- ex) [MRF<text style="color:lightgray;">(Markov random field)</text>](https://ko.wikipedia.org/wiki/조건부_무작위장)

### 좋은 모델 ? 데이터의 패턴을 잘 학습한 모델
- 여러 요건으로 살펴 볼 수 있다
	- [bias-variance trade-off<text style="color:lightgray;">(편향-분산 트레이드오프)</text>](https://ko.wikipedia.org/wiki/편향-분산_트레이드오프)
		- 모델이 더 나은 성능을 내려면 편향을 줄이거나 분산을 줄여야 함에 집중
		- 편향이 적당히 유연하면서 복잡도가 낮아 분산이 작게 나오는 모델
		- 방법
			- boosting<text style="color:lightgray;"> : 간단한 모델을 여러개 조합해 편향을 ↓</text>
			- random forest<text style="color:lightgray;"> : 복잡한 모델인 결정트리를 여러개 조합하여 분산을 ↓</text>
	- [regularization<text style="color:lightgray;">(정규화)</text>](http://astralworld58.tistory.com/64)
		- 정해진 모델이 필요 이상으로 복잡해지지 않도록 조절하는 트릭
		- 방법
			- 모델변경<text style="color:lightgray;"> : 데이터를 표현하는 방법을 새롭게 전환, 적합한 모델 탐색</text>
			- 정규화<text style="color:lightgray;"> : 모델에 들어있는 인자에 제한을 둠, 모델이 복잡해 지지 않게 하는 방법</text>
				- 모델의 표현식에 추가적인 제약조건을 걺, 필요이상으로 모델이 복잡해 지지 않도록함

## Loss function<text style="color:lightgray;">(손실함수)</text>
- 모델이 실제 데이터를 바르게 반영하는지, 얼마나 예측이 정확한지 수학적으로 표현하는 것
- loss function으로 얻은 값은 <text style="font-weight: bolder;">ERROR</text> <text style="color:lightgray;">: 값이 작을수록 모델이 더 정확하게 학습 된것</text>
- 대표종류 4가지
	- 산술 손실함수 : 모델로 산술값을 예측할 때, 데이터에 대한 예측값과 실제 관측값을 비교
	- 확률 손실함수 : 모델로 항목이나 값에 대한 확률을 예측하는 경우에 사용
		- 정답을 맞출 확률을 최대화
	- 랭킹 손실함수 : 모델로 순서를 결정할 때 사용
		- 결과값에 대한 손실을 측정하지 않음
		- 예측해낸 결과값의 순서가 맞는지만 판별함
	- 모델 복잡도와 관련된 손실함수 : 나머지 3가지 손실함수들과 합쳐져서 모델이 필요이상으로 복잡해지지 않도록 방지함

## 최적화 : 실제 학습을 하는 방법
- Loss function의 결과값을 최소화하는 모델의 인자를 찾는 것
- 대표방법
	- [Gradient descent(경사하강법)](http://gdyoon.tistory.com/9)
		- 임의의 지점에서 시작해서 경사를 따라 내려갈 수 없을 때까지 반복적으로 내려가며 최적화를 수행
			- 경사는 방향을 결정함, 경사는 곡선의 접선 방향으로 결정되고, 손실함수를 인자 ⍬에 대해 미분하여 구함
	- [Newton/quasi-newton method(뉴턴/준뉴턴 방법)](http://darkpgmr.tistory.com/58)
		- 뉴턴 방법
			- 임의의 학습률을 사용하는 대신 1차 gragien와 2차 gradient를 활용하여 업데이트함
				<text style="color:lightgray;">- 2차 gradient를 구하기 어려워서 잘 사용하지 않음</text>
		- 준뉴턴방법
			-  2차 gradient를 직접계산✖︎, 1차 gradient를 이용해 2차 값의 근사값을 구해서 사용함
			- ex) BFGS, LBFGS
				- 위 둘은 데이터가 너무 많지 않은 경우에 뛰어난 성능을 보임
	- stochastic gradient descent(확률적 경사하강법) - [경사하강법]((http://darkpgmr.tistory.com/133))
		- 일부 데이터만 이용해서 손실함수와 1차 gradient값을 근사적으로 계산하는 SGD가 만들어짐
		- n개의 샘플을 뽑아서 손실함수와 1차 gradient값을 계산함
		- SGD는 GD에 비해 gradient가 불안정해짐
			- 적은양의 데이터를 이용하므로
		- 시스템을 완성하기 전에 미니 배치의 크기와 학습률을 다양하게 시도해본 후 <text style="font-weight: bolder;">적절한 값을 선택</text>하는 것이 중요
	- [backpropagation(역전파)](https://brunch.co.kr/@chris-song/22)
		- 딥러닝에서 많이 사용하는 방식
		- 한쪽 방향으로 층층이 연결된 구조에서의 최적화를 효율적으로 수행하는 방식

## 모델 평가 : 성능을 평가하는 방법
- Generalization<text style="color:lightgray;">(일반화: 새로운 데이터가 들어왔을 때 잘 동작하는지 측정하는 것)</text>
- Generalization 평가 항목
	- 모델의 일반화 특성평가
		- 학습-평가 데이터나누기 → 교차검증 
	- 정확도
		- 모델이 데이터를 얼마나 정확하게 분류했는지에 대한 평가 지표
		- 전체 데이터 대비 분류 적확도를 구하는 것
	- Precision(정밀도)과 Recall(포괄성)
		- Positive & Negative를 결정하는 Binary classification(이진화 분류 문제)에서 정의됨
		- 모델이 양성으로 예측한 데이터 중에서 얼마나 정답을 맞혔는지 평가함
- 랭킹평가
	- 정밀도@K
		- 랭킹을 구한 후 앞에서부터 K번째까지의 결과중에서 몇개가 올바른지 검사
	- NDCG
		- 정밀도@K 검사와 유사함
		- 중요도를 순서에 따라 바꾸어가며 평가함
