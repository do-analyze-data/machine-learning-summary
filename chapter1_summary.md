# 머신러닝 기초 지식
- 머신러닝, 딥러닝

## 시작하기

- 머신러닝

	- 데이터를 이용해서 명시적으로 정의되지 않은 패턴을 컴퓨터로 학습하여 결과를 만들어 내는 학문
	- <text style="font-weight: bold;">데이터 + 패턴인식 + 컴퓨터를 이용한 계산</text>
    	- 테이터 = 머신러닝은 항상 데이터 기반
    	- 패턴인식 = 딥러닝을 이용하여 데이터의 패턴을 유추하는 방법이 주축
    		- <text style="color:orange;">사용자가 정해놓은 패턴 ✖︎</text>, 데이터를 보고 패턴을 추리하는 것
        - 컴퓨터를 이용한 계산 = 계산속도 ↑, 데이터 처리 효율성 ↑
	- 패턴 인식을 기반으로 컴퓨터를 이용해서 데이터를 처리하는 학문
- 머신러닝의 3가지 관점

| | 통찰력 | 이론적 엄정성 | 데이터 적합성 |
| ------ | ------ | ------ | ------ | 
| 선호하는 출발점 | 데이터에 대한 지식 | 수학적인 증명 기법, 잘 정의되고 보장된 모델 | 데이터에 잘 맞는 모델 |

## 머신러닝 분류
- 목표 분류
	- Supervised Learning (지도학습)
		- 값/레이블을 <text style="color:red;">예측</text>하는 시스템 구축 	
		- ex) 예측하기, 분류하기
	- Unsupervised Learning (비지도학습)
		- <text style="color:red;">패턴</text> 추출
		- ex) 군집화, 이상검출, 데이터 분포 추측
	- Reinforcement learning (강화학습)
		-  <text style="color:red;">상호작용</text> 가능한 시스템 구축
		-  각각 행동에 대한 피드백을 받아서 <text style="color:red;">다음 행동</text>을 정하는 알고리즘을 학습
- 기법 분류
	- 통계
	- 딥러닝

### Supervised의 분류
- Regression( 회귀 ) : 값 예측 
	- ex) 온도 변화 추이를 보고 온도 예측	
- Classification( 분류 ) : 항목선택
	- ex) 책의 카테고리 분류
- Ranking(랭킹)/ Recommend(추천) : 순서정렬
	- ex) 영화 추천

#### Recommend와 Regression의 차이
- Recommend : 상품에 대한 사용자의 선호도를 예측하는 시스템
- Regression : 하나의 입력으로 하나를 예측 X, 다양한 관계를 고려함

### Unsupervised의 분류
- Cluestering(군집화)/Topic Modeling
	: 유사데이터를 묶음
	- Cluestering
		- 비슷한 데이터를 묶어서 <text style="font-weight: bold;">큰단위로 만드는 기법</text>
		- 묶여진 그룹으로 데이터 패턴을 파악함
    - Topic Modeling
	    - 주로 텍스트 데이터에 사용, 관련 정도를 <text style="font-weight: bold;">Topic별 가중치를 사용해 확률로 표현</text>
- Density Estimation(밀도추정)
	: 데이터 분포 예측
    - 관측한 데이터로부터 데이터를 생성한 원래의 분포를 추측하는 방법
- Dimensionality Reduction(차원축소)
	: 데이터 차원 간추림
    - 높은 차원을 가진 데이터의 복잡도를 줄이기 위해 2,3차원으로 표현하기 위한 방법
    - ex) PDA(주성분분석), 특잇값 분해

## 딥러닝
- 신경망을 층층이 쌓아서 문제를 해결하는 기법의 총칭
- 사용하는 기법이 특정 형태를 가지는 것을 말함
- 특성 
	- 데이터 양에 의존
	- 문제에 대한 가정 ↓
	- 다양한 패턴과 경우에 유연하게 대응하는 구조를 만들어 많은 데이터를 이용하여 학습시키는 것으로 모델의 성능을 향상