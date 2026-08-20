# CTR 예측: 정형 피처와 행동 시퀀스를 함께 다룬 DCN-V2 실험

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## 한눈에 보기

광고 클릭 예측에서는 사용자·광고·시간대의 정형 정보와 사용자가 직전에 무엇을 봤는지 나타내는 행동 이력을 함께 읽어야 한다. 이 프로젝트는 DCN-V2 계열의 cross/deep 타워에 DIN·DIEN·BST 시퀀스 인코더를 교체 가능하게 붙여, 피처 상호작용과 관심의 시간적 흐름을 동시에 실험했다.

저장된 검증 메타데이터에서 DCN-V2+DIEN은 ROC-AUC 0.7413, PR-AUC 0.0792를 기록했다. DUSIN이라고 표기된 확장 실험은 ROC-AUC 0.7417, PR-AUC 0.0780이다. 두 지표의 최고값이 같은 모델에서 나오지 않았으므로, 한 모델이 절대적으로 우수하다고 쓰지 않고 지표별 결과를 그대로 남겼다.

## 해결하려 한 문제

클릭 여부는 희소한 이진 라벨이어서 정확도만으로 모델을 고르면 대부분을 ‘미클릭’으로 찍는 모델이 좋아 보일 수 있다. 여기서는 두 질문을 나눠 다뤘다.

1. 탭형 변수 사이의 비선형 조합을 어떻게 학습할 것인가?
2. 길이가 제각각인 과거 행동 이력을 어떻게 제한된 메모리 안에서 반영할 것인가?

첫 질문에는 DCN-V2 스타일의 CrossNetMix와 MLP 타워를 병렬로 사용했다. CrossNetMix는 저차원 행렬과 여러 expert의 게이팅으로 명시적인 교차항을 만들고, MLP는 더 일반적인 비선형 패턴을 담당한다. 두 출력을 합치지 않고 마지막 예측 헤드에서 시퀀스 관심 벡터와 함께 결합했다.

두 번째 질문에는 행동 이력을 최대 50개로 자르고, 아이템 ID를 262,144개 해시 버킷으로 보냈다. 원본 ID 종류가 커도 임베딩 크기를 통제하기 위한 선택이다. 짧은 이력은 왼쪽을 padding했고, 빈 시퀀스의 attention 계산에서는 `NaN`이 생기지 않도록 별도 처리했다.

## 데이터 처리와 학습 설계

원본 Parquet와 데이터 설명서는 저장소에 포함되지 않았으므로, 표본 수·출처·클릭 비율을 임의로 적지 않았다. 다만 실행 코드와 저장된 메타데이터로 확인할 수 있는 입력 구조는 다음과 같다.

- 라벨: 기본값 `clicked`
- 시퀀스: 기본값 `seq`; 대괄호·쉼표 형식과 숫자 문자열을 모두 파싱
- 식별자: 기본값 `ID`; 제출 파일에 보존
- 연속형: `float32` 변환 후 결측치는 0으로 처리
- 범주형: 학습 분할에서만 category-to-index 맵을 만들고, 검증·테스트에서 처음 보는 값은 OOV 인덱스로 처리

학습은 시드 42의 층화 85%/15% 분할을 사용한다. 양성·음성 개수 비율로 `BCEWithLogitsLoss(pos_weight=negative/positive)`를 구성해 불균형을 반영했고, AdamW·cosine annealing·ROC-AUC 기반 early stopping을 적용했다. 저장된 주요 실행은 전체 학습 데이터(`sample_subset=1.0`), batch size 512, 학습률 0.001, 최대 3 epoch, dropout 0.2 조건이다.

## 모델을 비교한 방식

시퀀스 인코더는 같은 DCN-V2 골격 안에서 교체했다. DIN은 현재 후보와 과거 아이템의 attention을 계산하고, DIEN은 임베딩 시퀀스를 GRU로 한 번 통과시켜 관심의 순서 변화를 반영한다. BST는 positional encoding과 Transformer encoder를 사용한다. 따라서 비교의 초점은 단순히 “딥러닝을 썼다”가 아니라, 어떤 이력 표현이 같은 탭형 피처 위에서 더 나은 순위를 만드는지에 있다.

| 저장된 검증 실행 | ROC-AUC | PR-AUC | 해석 |
|---|---:|---:|---|
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention 기반 기본 시퀀스 모델 |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | Transformer 기반 이력 인코더 |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | 저장된 실행 중 PR-AUC 최고 |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | 저장된 실행 중 ROC-AUC 최고 |

모델 결과는 `results/*.json`에 남아 있는 validation 결과다. 외부 테스트셋이나 리더보드 점수가 아니며, 서로 다른 seed 반복·신뢰구간·캘리브레이션 검증까지 수행한 결과도 아니다. PR-AUC가 가장 높은 DIEN을 기본 비교 기준으로 삼되, 운영 목표가 순위 구분인지 양성 후보 정밀도인지에 따라 선택이 달라질 수 있음을 문서에 남겼다.

## 구현에서 중요했던 선택

`src/train.py`는 전체 파이프라인의 기준 진입점이다. 데이터를 읽고, 학습/검증 분할을 만들고, 학습 전용 범주형 사전을 구성한 뒤, 검증 점수와 제출 CSV·메타 JSON을 저장한다. 이는 노트북의 여러 실험을 전부 자동화한 것은 아니지만, DIN·DIEN·BST 비교를 같은 명령행 인자로 재실행할 수 있게 만든 경로다.

다만 `main.py`의 LightGBM 피처 엔지니어링 스텁은 이 PyTorch 파이프라인과 별개다. 또한 DUSIN 실험 일부는 노트북에서만 남아 있어 현재 CLI의 DIN/DIEN/BST 선택지와 동등한 제품 경로라고 볼 수 없다. 포트폴리오에서는 이 차이를 숨기기보다, 실험 확장안과 유지되는 학습 경로를 구분했다.

## 다음에 보완할 부분

가장 먼저 필요한 것은 데이터 버전·행 수·양성 비율·분할별 건수·패키지 버전·사용 장비를 한 번에 적은 실험 매니페스트다. 그 다음에는 시간 순서 기반 평가, 반복 시드, calibration, Recall/Precision@K 같은 운영 지표를 추가해야 한다. 현재 결과는 시퀀스와 탭형 피처를 결합한 모델링 역량을 보여 주지만, 실제 광고 예산 의사결정 성과까지 입증하지는 않는다.

## 근거 파일

- 학습 파이프라인: [`src/train.py`](src/train.py)
- 데이터·시퀀스 처리: [`src/data.py`](src/data.py)
- DCN-V2 및 DIN/DIEN/BST 구현: [`src/models.py`](src/models.py)
- 결과 메타데이터: [`results/dcnv2_dien_meta.json`](results/dcnv2_dien_meta.json), [`results/dcnv2_dien_dusin_full_meta.json`](results/dcnv2_dien_dusin_full_meta.json)
- 실행 경계: [`research/RUN_MANIFEST.md`](research/RUN_MANIFEST.md)
