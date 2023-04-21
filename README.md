# ETRI 2023 휴먼이해 인공지능 경진대회

## Abstract

최근 센싱 기술과 인공지능 기술의 발전으로 멀티모달 데이터와 이를 이용한 감정 인식에 대한 연구가 활발히 진행되고 있다. 본 연구에서는 멀티모달 데이터를 이용한 Random Forest 감정 예측 모델을 제안한다. Extra Trees Classifier, Ridge Classifier, KNN 모델과 성능을 비교하였으며 센서 데이터, 레이블 데이터, 수면 데이터, 설문조사 데이터를 종합적으로 사용한 경우와 센서 데이터만 사용한 경우에 대해 모델 성능 차이를 조사하였다. 그 결과, 다른 모델에 비해 Random Forest의 성능이 우수했으며 단순히 센서 데이터만 사용할 때보다 여러 데이터를 사용할 때, 모델이 더 나은 성능을 가졌다. 추가로 Explainable Artificial Intelligence(XAI)를 이용하여 감정 인식에 사용되는 feature가 모델에 어떤 영향을 미치는지 확인하였다. 이를 통해 예측 모델 결과의 신뢰성을 증명하였다. 본 연구의 감정 예측 모델은 의료, 마케팅, 교육 등 다양한 분야에 활용될 수 있으며 기존의 감정 예측 모델을 개선하는데 이바지할 수 있다.

1. 소개
1.1 대회 소개
분야: 라이프로그 데이터셋 활용 인식 및 추론 기술 분야
활용 데이터: ETRI 라이프로그 데이터셋 활용 연구(ETRI 라이프로그 데이터셋(2020-2018))

1.2 진행 실험
- Random Forest와 KNN 등의 인공지능 모델을 활용하여 멀티모달 데이터를 바탕으로 사용자의 감정을 분석, 예측하는 연구를 진행한다.
- Explainable Artificial Intelligence(XAI) 기술을 사용하여 감정 예측에 사용되는 feature가 어떤 영향을 미치는지 뚜렷하게 파악하여 블랙박스 문제를 해결하고자 한다. 

(제안모델 프레임워크 삽입)

1.3 코드 설명
model.ipynb : 모델 성능 비교 실험 
shap.ipynb: XAI 실험

1.4 데이터 전처리

‘ETRI 라이프로그 데이터셋 (2020-2018)’은 멀티모달 센서를 활용한 라이프로그 데이터셋으로 센서 데이터, 라벨 데이터, 수면 데이터와 설문조사 데이터를 포함한다. 

1.4.1 센서 데이터

센서 데이터는 전기 피질 활성 데이터인 ‘e4Eda’와 심박수를 나타내는 ‘e4Hr’, 체온을 나타내는 ‘e4Temp’를 사용한다. 

1.4.2 레이블 데이터
레이블 데이터는 시간별 사용자의 행동, 상태, 기분 등을 담은 데이터이다.
이 중 ‘action’, ‘actionOption’, ‘actionSub’, ‘actionSubOption’은 이동 수단을 나타내는 ‘actionSubOption’과 어떤 종류의 행동을 했는지를 나타내는 ‘action’으로 통합하였고, 나머지 feature는 제거했다. ‘condition’, ‘conditionSub1Option’, ‘conditionSub2Option’은 누구와 있는지를 나타내는 feature인 ‘CONDITION’으로 통합하고 나머지는 삭제하였다. 

1.4.3 수면 데이터
‘startDt’와 ‘endDt’의 시간(hour)만을 가져와 잠든 시간을 나타내는 ‘startHour’와 일어난 시간을 나타내는 ‘endHour’라는 feature를 추가했다. 그리고 다른 데이터들과 통합시키기 위해 ‘year-month-day hour:minute:second’ 형식이던 ‘startDt’와 ‘endDt’를 ‘year-month-day’ 형태로 바꾼 ‘startDate’와 ‘endDate’를 추가했다. ‘endDate’가 중복되는 행에 대해서는 가장 수면 시간이 긴 것만을 남기고 제거했다.

1.4.4 설문조사 데이터
설문조사 데이터는 사용자가 직접 입력한 데이터로 하루에 대한 정보를 기입하여 만들어졌다. 해당 데이터는 연도별 ‘alcohol’의 단위에 차이가 있어 ‘alcohol’과 ‘aAmount’를 통합해 술을 마신 여부를 나타내는 ‘alcohol’로 만들었다. 이후 오전에 한 설문조사와 오후에 한 설문조사를 ‘inputDt’를 기준으로 합쳤고, ‘amPm’은 삭제하였다. 

1.4.5 데이터 통합
 센서 데이터, 라벨 데이터, 수면 데이터와 설문조사 데이터를 ‘ts’와 날짜를 기준으로 통합하였다. 센서 데이터와 레이블 데이터는 ‘ts’를 기준으로 inner join 했고, 수면 데이터와 설문조사 데이터는 각각 ‘endDate’와 ‘inputDt’를 기준으로 합쳤다. 이를 날짜를 기준으로 다시 합친 후, 날짜를 나타내는 feature는 중복되어 삭제하였다.

1.4.6 표준화 & one-hot encoding
 데이터셋의 feature 중, 입력 데이터에 속하는 24개의 feature에 대해, 범주형 데이터는 one-hot encoding 하였으며, 수치형 데이터는 표준화하였다.

2. 코드 실행 방법
2.1 데이터셋 다운로드
(우리가 전처리한 데이터.csv) 를 다운로드합니다.

2.2 모델 성능 비교 실험
모델 성능 비교 실험을 위해서 pycaret 라이브러리를 import 합니다.
    pip install pycaret

2.3 XAI 실험
XAI 실험을 위해서 shap 라이브러리를 import 합니다.
      pip install shap

3. 성능
3.1 모델별 성능비교 결과

(표 삽입)

3.2 Feature Importance 결과

(결과 사진 삽입)

4. Reference
[1] Haag, Andreas, et al. "Emotion recognition using bio-sensors: First steps towards an automatic system." Affective Dialogue Systems: Tutorial and Research Workshop, ADS 2004, Kloster Irsee, Germany, June 14-16, 2004. Proceedings. Springer Berlin Heidelberg, 2004.
[2]Seungeun Chung, Chi Yoon Jeong, Jeong Mook Lim, Jiyoun Lim, Kyoung Ju Noh, Gague Kim, Hyuntae Jeong,
Real-world multimodal lifelog dataset for human behavior study. ETRI Journal 43(6), 2021 
[3] Jenke, Robert, Angelika Peer, and Martin Buss. "Feature extraction and selection for emotion recognition from EEG." IEEE Transactions on Affective computing 5.3 327-339, 2014
[4] Breiman, Leo. "Random forests." Machine learning 45 5-32, 2001
