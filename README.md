# ETRI 2023 휴먼이해 인공지능 경진대회
본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다

## Abstract
최근 센싱 기술과 인공지능 기술의 발전으로 멀티모달 데이터와 이를 이용한 감정 인식에 대한 연구가 활발히 진행되고 있다. 본 연구에서는 멀티모달 데이터를 이용한 Random Forest 감정 예측 모델을 제안한다. Extra Trees Classifier, Decision Tree, KNN 모델과 성능을 비교하였으며 센서 데이터, 레이블 데이터, 수면 데이터, 설문조사 데이터를 종합적으로 사용한 경우와 센서 데이터만 사용한 경우에 대해 모델 성능 차이를 조사하였다. 그 결과, 다른 모델에 비해 Random Forest의 성능이 우수했으며 단순히 센서 데이터만 사용할 때보다 여러 데이터를 사용할 때, 모델이 더 나은 성능을 가졌다. 추가로 Explainable Artificial Intelligence(XAI)를 이용하여 감정 인식에 사용되는 feature가 모델에 어떤 영향을 미치는지 확인하였다. 이를 통해 예측 모델 결과의 신뢰성을 증명하였다. 본 연구의 감정 예측 모델은 의료, 마케팅, 교육 등 다양한 분야에 활용될 수 있으며 기존의 감정 예측 모델을 개선하는데 이바지할 수 있다.

## 1. 소개
### 1.1 대회 소개
**분야**

라이프로그 데이터셋 활용 인식 및 추론 기술 분야

**활용 데이터**

ETRI 라이프로그 데이터셋 활용 연구(ETRI 라이프로그 데이터셋(2020-2018))

### 1.2 진행 실험
- Random Forest와 KNN 등의 인공지능 모델을 활용하여 멀티모달 데이터를 바탕으로 사용자의 감정을 분석, 예측하는 연구를 진행한다.
- Explainable Artificial Intelligence(XAI) 기술을 사용하여 감정 예측에 사용되는 feature가 어떤 영향을 미치는지 뚜렷하게 파악하여 블랙박스 문제를 해결하고자 한다. 

<img src="https://user-images.githubusercontent.com/90269177/233654468-8e6e42b1-5355-47b9-8971-b1228a30b459.png" width="473" height="526">

### 1.3 코드 설명
- Main.ipynb : main 함수
- preprocessing.py : 데이터셋 전처리
- model.py : 모델 성능 비교 실험 
- xai.py : XAI 실험
- utils : 실험에 필요한 함수 모음

### 1.4 데이터 전처리
‘ETRI 라이프로그 데이터셋 (2020-2018)’은 멀티모달 센서를 활용한 라이프로그 데이터셋으로 센서 데이터, 라벨 데이터, 수면 데이터와 설문조사 데이터를 포함한다. 

#### 1.4.1 센서 데이터
센서 데이터는 전기 피질 활성 데이터인 ‘e4Eda’와 심박수를 나타내는 ‘e4Hr’, 체온을 나타내는 ‘e4Temp’를 사용한다. 

#### 1.4.2 레이블 데이터
레이블 데이터는 시간별 사용자의 행동, 상태, 기분 등을 담은 데이터이다.
이 중 ‘action’, ‘actionOption’, ‘actionSub’, ‘actionSubOption’은 이동 수단을 나타내는 ‘actionSubOption’과 어떤 종류의 행동을 했는지를 나타내는 ‘action’으로 통합하였고, 나머지 feature는 제거했다. ‘condition’, ‘conditionSub1Option’, ‘conditionSub2Option’은 누구와 있는지를 나타내는 feature인 ‘CONDITION’으로 통합하고 나머지는 삭제하였다. 

#### 1.4.3 수면 데이터
‘startDt’와 ‘endDt’의 시간(hour)만을 가져와 잠든 시간을 나타내는 ‘startHour’와 일어난 시간을 나타내는 ‘endHour’라는 feature를 추가했다. 그리고 다른 데이터들과 통합시키기 위해 ‘year-month-day hour:minute:second’ 형식이던 ‘startDt’와 ‘endDt’를 ‘year-month-day’ 형태로 바꾼 ‘startDate’와 ‘endDate’를 추가했다. ‘endDate’가 중복되는 행에 대해서는 가장 수면 시간이 긴 것만을 남기고 제거했다.

#### 1.4.4 설문조사 데이터
설문조사 데이터는 사용자가 직접 입력한 데이터로 하루에 대한 정보를 기입하여 만들어졌다. 해당 데이터는 연도별 ‘alcohol’의 단위에 차이가 있어 ‘alcohol’과 ‘aAmount’를 통합해 술을 마신 여부를 나타내는 ‘alcohol’로 만들었다. 이후 오전에 한 설문조사와 오후에 한 설문조사를 ‘inputDt’를 기준으로 합쳤고, ‘amPm’은 삭제하였다. 

#### 1.4.5 데이터 통합
센서 데이터, 라벨 데이터, 수면 데이터와 설문조사 데이터를 ‘ts’와 날짜를 기준으로 통합하였다. 센서 데이터와 레이블 데이터는 ‘ts’를 기준으로 inner join 했고, 수면 데이터와 설문조사 데이터는 각각 ‘endDate’와 ‘inputDt’를 기준으로 합쳤다. 이를 날짜를 기준으로 다시 합친 후, 날짜를 나타내는 feature는 중복되어 삭제하였다.

#### 1.4.6 표준화 & one-hot encoding
데이터셋의 feature 중, 입력 데이터에 속하는 24개의 feature에 대해, 범주형 데이터는 one-hot encoding 하였으며, 수치형 데이터는 표준화하였다.

## 2. 코드 실행 방법
### 2.1 데이터셋 다운로드

ETRI 라이프로그 데이터셋(2020-2018)을 다운로드한다.

링크: <https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR>

데이터 파일명
* 2018 data
* 2019 data
* user01-06 data
* user07-10 data
* user11-12 data
* user21-25 data
* user26-30 data
* 2019-2018 수면 측정 데이터
* 2020 수면 측정 데이터
* 2019-2018 수면관련 설문결과
* 2020 수면관련 설문결과


### 2.2 모델 성능 비교 실험
모델 성능 비교 실험을 위해서 pycaret 라이브러리를 설치한다. 

    pip install pycaret

### 2.3 XAI 실험
XAI 실험을 위해서 shap 라이브러리를 설치한다.

    pip install shap

### 2.4 Main.py 실행
업로드 된 모든 파이썬 파일을 다운로드한 후, Main.py를 실행하여 모든 실험 결과를 확인한다.

## 3. 성능
### 3.1 모델별 성능비교 결과
#### 3.1.1 emotionPositive 예측 모델 성능 비교

| 모델명                 | accuracy   | recall     | precision  | F1-score |
| ---------------------- | ---------- | ---------- | ---------- | -------- |
|KNN                     | 0.8161     | 0.8399     | 0.8412     | 0.8406   |
| **Random Forest**      | **0.8700** | **0.8868** | **0.8919** |**0.8893**|
| Extra Trees Classifier | 0.8610     | 0.8799     | 0.8868     | 0.8833   |
| Decision Tree          | 0.8261     | 0.8454     | 0.8556     | 0.8505   |

#### 3.1.2 emotionTension 예측 모델 성능 비교

| 모델명                 | accuracy   | recall     | precision  | F1-score |
| ---------------------- | ---------- | ---------- | ---------- | -------- |
|KNN                     | 0.7758     | 0.7767     | 0.7729     | 0.7746   |
| **Random Forest**      | **0.8488** | **0.8483** | **0.8455** |**0.8468**|
| Extra Trees Classifier | 0.8352     | 0.8345     | 0.8322     | 0.8333   |
| Decision Tree          | 0.7978     | 0.7964     | 0.7947     | 0.7955   |

#### 3.1.3 센서 데이터만 사용하여 emotionPositive 예측 모델 성능 비교

| 모델명                 | accuracy   | recall     | precision  | F1-score |
| ---------------------- | ---------- | ---------- | ---------- | -------- |
|KNN                     | 0.2843     | 0.1903     | 0.2610     | 0.2137   |
| **Random Forest**      | **0.2867** | **0.1792** | **0.2640** |**0.1828**|
| Extra Trees Classifier | 0.2832     | 0.0342     | 0.6007     | 0.0587   |
| Decision Tree          | 0.2907     | 0.1377     | 0.2930     | 0.1605   |

#### 3.1.4 센서 데이터만 사용하여 emotionTension 예측 모델 성능 비교

| 모델명                 | accuracy   | recall     | precision  | F1-score |
| ---------------------- | ---------- | ---------- | ---------- | -------- |
|KNN                     | 0.2203     | 0.2262     | 0.2231     | 0.2194   |
| **Random Forest**      | **0.2531** | **0.2736** | **0.2340** |**0.1953**|
| Extra Trees Classifier | 0.2471     | 0.2689     | 0.3013     | 0.1202   |
| Decision Tree          | 0.2492     | 0.2663     | 0.2439     | 0.2387   |

### 3.2 Feature Importance 결과
첫번째 사진은 emotionPositive와 emotionTension의 전체적인 feature importance를 나타낸 것이다.
나머지 사진들은 emotionPositive가 1-5의 값을 가질 때 emotionTension가 1-7의 값을 가질 때 shap value를 시각화한 결과이다.

shap value 시각화에서 나타난 붉은색과 파란색은 해당 변수의 값이 높은 경우와 낮은 경우를 나타낸다. 붉은색으로 표시된 부분은 해당 변수의 값이 높아질 수록 해당 y_value의 출력값이 높아지는 경향을 보이며, 파란색으로 표시된 부분은 해당 변수의 값이 낮아질수록 출력값이 높아지는 경향을 미친다.

#### 3.2.1 emotionPositive
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233658640-e1c69efe-ade0-4052-9491-4ca0375b9129.png" width="473" height="526">
</figure>

- emotionPositive가 1일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233662886-48690e26-2c99-42dd-a9b3-8649f2da0b7b.png" width="473" height="526">
</figure>

- emotionPositive가 2일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233662924-8f812cf3-4c79-455f-9464-4f75a7468a44.png" width="473" height="526">
</figure>

- emotionPositive가 3일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233662977-78a0c5b0-0598-4acf-b5c1-dcdf32df5258.png" width="473" height="526">
</figure>

- emotionPositive가 4일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233663017-0d8aec8f-c09f-4206-ac41-54e5032dab9e.png" width="473" height="526">
</figure>

- emotionPositive가 5일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233664307-339bb06d-c8af-4e94-a9c2-fb0927ee6997.png" width="473" height="526">
</figure>

#### 3.2.2 emotionTension
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233658878-fb210050-30a7-4e23-80f3-469915c4c1b3.png" width="473" height="526">
</figure>

- emotionTension이 1일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233668980-334a605a-a616-4529-bbbc-8e55cfb7ee18.png" width="473" height="526">
</figure>

- emotionTension이 2일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233668992-87bf71ec-e6cd-4e21-9544-6cf5b4521b9f.png" width="473" height="526">
</figure>

- emotionTension이 3일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233669014-0109e60d-7794-4d5b-af9b-7a89c99b7848.png" width="473" height="526">
</figure>

- emotionTension이 4일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233669038-ccb10ca7-3edc-442b-89dd-90cf90b34957.png" width="473" height="526">
</figure>

- emotionTension가 5일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233669417-56341894-8da6-4803-b35c-ed490c21a640.png" width="473" height="526">
</figure>

- emotionTension이 6일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233669055-47d4c6a9-015e-4d35-87f6-af7e760e5282.png" width="473" height="526">
</figure>

- emotionTension이 7일 때 SHAP_value
<figure>
    <img src="https://user-images.githubusercontent.com/90269177/233669072-17de746a-6bfd-4e21-b58a-1915d9f30752.png" width="473" height="526">
</figure>

## 4. Reference
[1] Haag, Andreas, et al. "Emotion recognition using bio-sensors: First steps towards an automatic system." Affective Dialogue Systems: Tutorial and Research Workshop, ADS 2004, Kloster Irsee, Germany, June 14-16, 2004. Proceedings. Springer Berlin Heidelberg, 2004.

[2] Seungeun Chung, Chi Yoon Jeong, Jeong Mook Lim, Jiyoun Lim, Kyoung Ju Noh, Gague Kim, Hyuntae Jeong,
Real-world multimodal lifelog dataset for human behavior study. ETRI Journal 43(6), 2021 

[3] Jenke, Robert, Angelika Peer, and Martin Buss. "Feature extraction and selection for emotion recognition from EEG." IEEE Transactions on Affective computing 5.3 327-339, 2014

[4] Breiman, Leo. "Random forests." Machine learning 45 5-32, 2001
