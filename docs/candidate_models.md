# Candidate Models — 실험 카탈로그

향후 실험 후보 정리. 각 모델: **1-2줄 설명 + 라이브러리 + 우리 데이터 적합도**. EDA 인사이트 ([docs/eda_findings.md](eda_findings.md)) 기반 우선순위 표시.

---

## 0. 진행 상태

| exp | model | self-val NDCG@10 | self-val recall@10 | public NDCG@10 | self-val/public | status |
|---|---|---:|---:|---:|---:|---|
| 000 | ALS (baseline match: factors=32, alpha=10, reg=0.001, weights 1/1/1, filter=False) | 0.1838 | 0.2558 | **0.0791** | 2.32 | ✅ 완료 |

**Calibration 확정** (2026-05-20):
- 우리 public 0.0791 vs baseline 공시 0.0847 = 6.6% 갭 → 학습 데이터 7일 손실(5.8%)과 거의 일치 → 파이프라인 정상
- **self-val ÷ public ≈ 2.32** → 향후 모델 비교 시 self-val에서 ÷2.32 환산해 public 기대치 추정 (같은 split 가정)
- 단, 이 비율은 ALS + Feb 23-29 hold-out 조합 측정값. 시퀀셜 모델은 비율 다를 수 있어서 첫 시퀀셜 끝나면 re-calibration 권장

---

## 1. EDA 강력 권장 (★★★)

| 모델 | family | 라이브러리 | 한 줄 설명 | 우리 데이터 적합 근거 |
|---|---|---|---|---|
| **TiSASRec** | Sequential (Transformer) | RecBole | SASRec + **time-interval embedding** — 두 이벤트 사이 시간 차를 attention key에 직접 주입 | time gap CV=4.12 → 시간 간격이 매우 불규칙. 시간 무시하면 정보 손실 |
| **BSARec** (AAAI 2024) | Sequential (Transformer) | 비공식 PyTorch impl (GitHub) | self-attention의 약점(노이즈 민감)을 frequency-domain inductive bias로 보완 | 최신 SOTA. literature상 NDCG +10~14%. RecBole 호환 가능 (래퍼 작성 필요) |
| **FEARec** | Sequential (Transformer) | RecBole | Frequency-Enhanced Attention (FFT 기반) — long-range pattern 강화 | 119k user sequence에 long-pattern 존재 가능 (4개월 데이터) |
| **SAFERec** | NBR + Sequential | 비공식 | Frequency-aware re-ranker — 빈도 신호 직접 모델링 | 우리 데이터 item 반복도 14.63% → NBR 시그널 작동 구간 |
| **MB-STR** (multi-behavior SASRec) | Sequential (Transformer) | 비공식 (논문 기반 구현) | SASRec + **behavior-type embedding** (view/cart/purchase as 0/1/2). item embedding + position embedding + behavior embedding 합산. cart/purchase position에 loss weighting | 우리 데이터가 정확히 3-behavior 구조 (view 99.78% / cart 0.20% / purchase 0.02%). 단일 SASRec은 event_type 정보를 버리지만 MB-STR은 직접 시그널화 — 적합도 매우 높음 |

---

## 2. EDA 보조 권장 (★★☆)

| 모델 | family | 라이브러리 | 한 줄 설명 | 우리 데이터 적합 근거 |
|---|---|---|---|---|
| **TIFU-KNN** | NBR | 비공식 (가벼움) | Temporal Item Frequency 기반 KNN — 최근 산 게 또 사고 싶다 (basket reco) | 반복도 14.6% + cart→purchase 0.8% (10x view→purchase) — 의도 강한 시그널 |
| EASE / EASER | Item-item | 직접 구현 쉬움 (~30줄) | item × item ridge regression closed-form. 학습 = single matrix inverse | sparse implicit feedback에서 강력. 학습 빠름, 디버깅 쉬움 |
| LightGCN | Graph CF | RecBole | user-item bipartite graph의 GCN — embedding propagation | implicit feedback CF의 강한 baseline. 우리처럼 view 99% 데이터에서 잘 작동 보고됨 |

---

## 3. EDA 후순위 (★☆☆)

| 모델 | 후순위 사유 |
|---|---|
| Mamba4Rec | 롱시퀀스 (>50) user 4.1%만 → 효율 우월성 발현 어려움 |
| CL4SRec | contrastive learning, 노이즈 대처 강하지만 우리 데이터 노이즈 평이 |

---

## 4. EDA가 다루지 않은 추가 후보

### Matrix Factorization / CF 변형

| 모델 | 라이브러리 | 한 줄 설명 | 우리 데이터 적용성 |
|---|---|---|---|
| BPR-MF | implicit | Pairwise ranking loss (positive > sampled negative) | ALS 대안. 학습 빠름. self-val에서 ALS와 비슷 또는 약간 낮을 가능성 |
| WRMF / iALS | implicit | Weighted ALS (Hu et al. 2008) — 우리가 쓴 그것 | ALS와 동일 family. 가중치 다양화 가능 |
| LMF (Logistic MF) | implicit | logistic loss 기반 MF | implicit feedback에서 ALS와 경쟁 가능 |

### Item-item / Neighbor

| 모델 | 라이브러리 | 한 줄 설명 |
|---|---|---|
| Item-item KNN | implicit | 코사인 유사도 / conditional probability 기반 item-item | 가장 단순. ensemble 입력으로 유용 |
| SLIM / SLIM-elastic | 직접 구현 또는 RecBole | sparse linear item-item (ElasticNet 정규화) | EASE의 sparse 버전. 학습 느림 |
| Item2Vec / Prod2Vec | gensim | word2vec on user session sequences | session sequence 활용, EDA의 user_session 데이터 자연스럽게 사용 |
| 공출현 (co-visitation) 룰 | 직접 구현 | "A를 본 사람이 B도 본다" 규칙 + score | 빠르고 직관적. ensemble 다양성 |

### Sequential 비-Transformer

| 모델 | 라이브러리 | 한 줄 설명 |
|---|---|---|
| **GRU4Rec** | RecBole | RNN 기반 session recommendation (2016년 고전) | 우리 시퀀스 짧음 (p90=29)이라 RNN으로 충분할 수도 |
| BERT4Rec | RecBole | Masked LM (BERT style) sequential rec | SASRec과 사촌. attention 양방향 |
| Caser | RecBole | CNN으로 (vertical=union, horizontal=skip) 패턴 학습 | 시퀀스 짧을 때 효과적. SASRec 대비 가벼움 |
| NARM | RecBole | Attention 기반 RNN — session intent 모델링 | session 정의 명확하면 강함 |
| STAMP | RecBole | short-term attention/memory priority | recent 행동 강조 — Feb spike 같은 burst 패턴에 강할 수 있음 |
| HGN / HSTU | 직접 구현 | hierarchical/long sequence transformer | 우리 데이터에선 overkill 가능성 |

### Graph

| 모델 | 라이브러리 | 한 줄 설명 |
|---|---|---|
| NGCF | RecBole | LightGCN의 prior — non-linear propagation 포함 | LightGCN보다 무거움, 보통 LightGCN 이김 |
| UltraGCN | RecBole | message passing 없이 constraint 기반 학습 — 빠름 | LightGCN보다 학습 10배 빠르고 비슷한 점수 |
| **SR-GNN** | RecBole | Session-aware graph (세션 내 item을 그래프 노드로) | 우리 user_session 컬럼 직접 활용. EDA 세션 분석 활용 가능 |

### Generative / VAE

| 모델 | 라이브러리 | 한 줄 설명 |
|---|---|---|
| Mult-VAE | RecBole | multinomial likelihood VAE — top-N rec의 강한 베이스라인 | 대규모 user × item에 적합. memory 주의 |
| RecVAE | RecBole | Mult-VAE 개선 (composite prior + adaptive denoising) | Mult-VAE의 진화형, 보통 더 잘함 |
| EASE보다 강함? | — | EASE가 더 자주 쓰임. VAE는 메모리 부담 |

### Two-stage / Reranker

| 단계 | 모델 후보 |
|---|---|
| **Stage 1 (candidate gen, ~200-500개)** | ALS, EASE, item2vec, co-visit, popularity hybrid |
| **Stage 2 (reranker, top-10)** | **LightGBM (LambdaMART)**, CatBoost, XGBoost — pairwise/listwise loss + handcrafted features |
| 특징 입력 예시 | user-item history feature (recency, frequency, brand match, price band, category match), candidate score from stage 1, time-of-day, day-of-week |

★ Two-stage는 점수 점프 잠재력이 큼 (kaggle competition에서 흔히 sequential 단독 대비 +5~15%). 단점: 복잡도/디버깅 비용 증가.

### Ensemble

| 방법 | 설명 |
|---|---|
| **Reciprocal Rank Fusion (RRF)** | 각 모델의 rank를 1/(k + rank) 점수로 변환, 합산. k=60 default. 점수 분포 다른 모델 합치기에 유리 |
| 가중 평균 (rank or score) | 모델별 weight (EDA 권고 0.35/0.30/0.20/0.15) 적용 |
| Borda count | rank 역수 합산 — RRF 단순화 버전 |
| Stacking | meta-learner (간단한 LR or LightGBM)이 모델 출력을 입력으로 받음. 단, 별도 holdout 필요 |

---

## 5. 최종 실험 plan — Week 1 모델링 + Week 2 서비스 (2026-05-21 확정)

### 목표 (재정의)

- **대회 NDCG 1% 짜내기 < 추천 시스템 대표 모델 경험 + 향후 서비스 개발 foundation + 포트폴리오 가치**
- 산업 ML 영역 (LightGBM/XGBoost ranker, 무거운 feature engineering, 전환율 학습 등) 은 본 단계에서 제외 — 다른 프로젝트에서 학습 가능
- 모델링 1주 + 서비스 개발 1주 분할

### Week 1 — 모델링 (5개 모델 × 4 paradigm)

| Day | 실험 | paradigm | 가치 |
|---|---|---|---|
| 1 | **EASE 직접 구현** (~30줄) | Item-item closed-form (classical) | foundation + 서비스 fast inference |
| 2-3 | **BSARec port** (저자 GitHub paper-to-code) | Sequential — frequency-domain | ★ AAAI 2024 SOTA, 포트폴리오 정점 |
| 4-5 | **DiffRec port** (저자 GitHub paper-to-code) | Diffusion (generative) | ★ SIGIR 2023, paradigm 차별화 |
| 6 | **LightGCN via RecBole** | Graph CF (mature) | classical-vs-modern 비교 baseline |
| 7 | **RRF Ensemble** + 제출 + 서비스 인터페이스 정리 | — | Week 2 준비 |

ALS (exp_000 완료) 까지 합치면 **6개 모델, 5 paradigm** (MF / Item-item / Sequential modern / Diffusion / Graph CF).

### Week 2 — 서비스 개발

- Week 1 의 모든 모델을 통일 `predict_for_user(user_id, top_k)` 인터페이스로 wrap
- FastAPI / Flask 등으로 API 서빙
- 실제 호출 → 추천 결과 출력 데모

### 의도적 제외 (한 줄씩)

| 제외 항목 | 이유 |
|---|---|
| LightGBM / XGBoost ranker, 무거운 FE | ML 영역, 우리 목표 (recsys core) 와 결이 다름 |
| SASRec / TiSASRec / FEARec / SAFERec / MB-STR / TIFU-KNN / CL4SRec | 외부 reference 모델 list 와 중복 (포트폴리오 차별화 ↓) |
| SR-GNN, NARM session 모드 | cross-session 94.6% — session-aware 모델 부적합 (EDA §15.3) |
| Mamba4Rec | 롱시퀀스 user 4.1%만 — 효율 우위 발현 어려움 (EDA §4) |
| Brand text embedding (word2vec init 등) | brand-category anomaly 영향 |
| LLM4Rec (TallRec, P5 등) | 인프라 부담, 1주 안에 무리 |

### 포트폴리오 강조점 (면접 talking points)

1. "EDA 인사이트 (time-irregular sequence, CV=4.12) 를 모델 선택 근거로 사용 — frequency-domain 접근 (BSARec)"
2. "BSARec (AAAI 2024) 저자 GitHub 의 PyTorch 구현을 우리 데이터에 맞게 port"
3. "DiffRec (SIGIR 2023) 으로 collaborative filtering 을 diffusion model paradigm 으로 접근"
4. "5 paradigm (MF / item-item / Sequential modern / Diffusion / Graph CF) 의 RRF ensemble"
5. "1주 만에 모델 5개 → 다음 주에 FastAPI 로 실제 서비스화"

### 보존된 이전 의사결정

- `filter_already_liked=False` (exp_000 lesson, 14.6% 반복도)
- max_len=50 (EDA p90=29)
- time-based holdout (val_days=7 또는 14)
- 예측 대상은 638,257 전원 (popularity fallback)

⚠️ **주의**: 자체 val 절대값을 절대 기준으로 삼지 말 것. exp_000이 self-val 0.18 (baseline 공시 public 0.08의 2.17배) 찍은 건 Feb 27~29 spike 효과. 다음 모델들의 self-val 을 ALS 0.18과 직접 비교 X, 각 모델의 ranking + recall@10 등 보조 메트릭 + (가능하면) public 제출 점수로 종합 판단.

---

## 6. 구현 비용 / 라이브러리 매트릭스

| 카테고리 | 라이브러리 즉시 사용 | 직접 구현 필요 | 비고 |
|---|---|---|---|
| MF (ALS/BPR) | implicit | — | exp_000에서 검증됨 |
| Sequential (SASRec/BERT4Rec/GRU4Rec/Caser/NARM/STAMP/TiSASRec/FEARec) | **RecBole** | — | 베이스라인 SASRec 코드 변형 가능 |
| Graph (LightGCN/NGCF/UltraGCN) | RecBole | — | implicit feedback에 native |
| EASE / SLIM | — (가벼우니) | scipy + sklearn으로 직접 (~30줄) | closed-form |
| Item2Vec | gensim | — | 빠르게 PoC 가능 |
| Co-visitation | — | pandas로 직접 (~50줄) | ensemble 다양성용 |
| **BSARec** (AAAI 2024, §5 채택) | — | 저자 GitHub PyTorch 구현 port (3일) | frequency-domain SOTA |
| **DiffRec** (SIGIR 2023, §5 채택) | — | 저자 GitHub PyTorch 구현 port (2일) | Diffusion paradigm 추천 시스템 |
| **MB-STR** | — | 비공식 (SASRec + behavior embedding ~50줄) | (§5에서 제외 — 외부 reference 중복) |
| **TIFU-KNN** | — | 비공식 (논문 기준 구현) | (§5에서 제외 — 외부 reference 중복) |
| Reranker (LightGBM) | lightgbm | feature engineering 필요 | (§5에서 제외 — ML 영역) |

서버 미리 설치된 것: `recbole, kmeans_pytorch, ray, implicit, pyarrow, fastparquet, tqdm` (per CLAUDE.md).

---

## 7. 운영진 팁 — 우리 데이터로 재검토

운영진 자료 ("순차 추천시스템 데이터 구성", "nDCG 지표 이해") 는 **일반적 sequential rec 입문 가이드**. 우리 경쟁 특성 (commerce, GT=purchase 0.02%, view 99.78%, Feb 27-29 spike, item 반복도 14.6%, 3-behavior) 에는 그대로 적용 불가. 항목별로 our-data fit을 비판적으로 검토.

### 7.1 Sequence 길이 — 일반론 적용 가능 ✓

| 항목 | 우리 default | 근거 |
|---|---|---|
| `max_len` | **50** | EDA p90=29 → 90%+ 커버 |
| Padding | left-pad 0 | 표준 |
| Slicing | right-keep 최신 | 표준 |
| Min length filter (**학습 한정**) | ≥3 events | sequence signal 부족 user 제외 |
| **예측 대상** | **638,257 전원** | 짧은 seq / cold-start user는 popularity_fallback (`core/submission.py` 구현됨) |

⚠️ **주의**: 우리 seq는 view가 99.78%. max_len=50 이 실질적으로 "최근 view 50개"가 됨. 모델 설정 시 sequence에 어떤 event_type 포함하는지 명시 (모두 / cart+purchase only / 가중 샘플링 등).

### 7.2 Train/Val Split — 운영진 leave-one-last 대신 time-based 유지

운영진 예시(leave-one-last)는 academic 논문 reproduce 표준. 우리 경쟁 setup과 정합 X:
- 경쟁 eval: Mar 1-7 **fixed time window**, user당 10개 multi-item
- leave-one-last: user당 last 1개 hold-out — 평가 단위가 다름

→ **time-based holdout 유지** (`core/validation.time_based_split`, Feb 23-29 val). exp_000 calibration ratio 2.32 재사용 가능.

학습 시그널은 별개: 각 user의 training-time sequence 위에 sliding window로 sequence-to-next 학습 (SASRec 표준 그대로). **train signal = sliding window** / **eval split = time-based** 분리.

### 7.3 Negative Sampling — 운영진 "uniform 1:1" 그대로 쓰면 손해

운영진 tip: "user history에 없는 item을 random uniform로 negative". **일반 movie/music 데이터에는 맞지만 우리 commerce + purchase GT 에는 신호 손실 큼**:

**왜 문제인가**:
- 우리 GT = purchase only. 학습은 "next event = purchase"를 잡아야 함
- user history 99.78%가 view → "history에 없는 item" = 진짜 무관심한 cold item (너무 쉬운 negative)
- 정작 학습 가치 큰 negative = **거의 살 뻔했지만 안 산 item** = user history **내부** (cart-not-purchased, view-not-purchased)
- Uniform random은 이걸 못 잡음

**우리 data에 맞는 변형 (ablation 가치 큼)**:

| 변형 | 설명 | 우리 task에서의 가치 |
|---|---|---|
| Uniform random (운영진 표준) | history 밖 item | baseline용, 학습 빠름 |
| **Hard negative — cart-not-purchased** | 같은 user가 cart 담았지만 안 산 item | "구매 직전 망설인" 신호 — 가장 강력 |
| **View-not-purchased** | 같은 user가 봤지만 안 산 item | 중간 난이도, sample 양 풍부 |
| Popularity-weighted | 인기 item을 더 자주 sampling | popularity bias 견디기 |
| In-batch negatives | 다른 user positive를 negative로 | 대규모 효율, SimCSE 스타일 |

**권장 전략**: 첫 sequential 실험은 uniform로 baseline 빠르게 → **두 번째 실험에서 cart-as-hard-negative ablation**. 잠재 lift 큼.

비율 1:1도 운영진 default일 뿐. 우리 GT가 sparse 하니 4:1 / 8:1 도 시도 가치.

### 7.4 NDCG@10 공식 — 구현 검증 완료 ✅

운영진 tip 공식: `rel / log2(i+2)` (i 0-indexed)
우리 `core/metrics.py:_discounts`: `1 / log2(arange(2, k+2))` → 수식 동일.

Binary relevance (purchase=1 / 그 외=0) 사용 — 경쟁 setup과 일치.

### 7.5 Tip 이 다루지 않은 우리-data-specific 이슈

운영진 자료는 generic intro라 commerce + purchase 예측 특성을 다루지 않음. **우리 plan에 추가로 들어가야 할 고려사항**:

| 이슈 | 우리 data 특성 | 모델링 함의 |
|---|---|---|
| **GT type vs train signal mismatch** | GT=purchase only (2,076건), train events 전체 8.35M | 옵션 A: train on next-any-event, eval on purchase / B: train on next-purchase only (signal 너무 적음, 학습 불가) → **A 채택 + §7.3 hard negative + 아래 loss weighting** |
| **View dominance (99.78%)** | 일반 SASRec는 모든 event_type 동등 처리 | MB-STR의 cart/purchase loss weighting 이 자연 정합. event_type embedding 추가 |
| **Feb 27-29 spike (69% of purchases)** | 학습 signal이 3일에 집중 | 학습 포함은 OK (전체 비중 큼). self-val 윈도우에 spike 포함되면 over-optimistic (exp_000 self-val 0.184 ↔ public 0.079 ratio 2.32의 절반 정도가 spike 효과로 추정). Feb 9-22 alternative split 검토 |
| **filter_already_liked=False** (exp_000 lesson) | item 반복도 14.6% — 본 것 다시 사는 패턴 강함 | Sequential 모델 inference 시 already-seen filter 끄기. 대부분 SASRec impl 기본값이 False지만 yaml 명시 |
| **Multi-behavior 정보 활용** | view/cart/purchase 3 type 존재 | 일반 SASRec는 정보 손실. MB-STR / event-type embedding / behavior-aware attention 가치 큼 (§1 ★★★ 후보) |
| **User demographic feature 없음** | 스키마에 연령/성별/지역 등 부재 ([eda_findings §1](eda_findings.md)) | reranker stage 2의 user-side feature는 **behavioral aggregate만** 가능 (top-brand, top-category, 평균 price, session 패턴). demographic-based personalization 불가 — 운영진 tip의 "연령/성별/지역 FE" 항목 직접 적용 불가 |
| **단기 시간 패턴 (요일/시간대)** | 4개월 winter window라 계절성 없음. 요일·시간대 효과는 EDA 미분석 ([eda_findings §13](eda_findings.md)) | Two-stage reranker feature 후보: 요일 (cyclic encoding), 시간대, day-of-month 등. 실제 reranker 직전에 EDA 보강 필요 |

### 7.6 Tip 의 NDCG 향상 lever ↔ 우리 plan 매핑

| Tip 권고 (일반론) | 우리 plan 대응 (구체) |
|---|---|
| 개인화 강화 | MB-STR (behavior signal), TiSASRec (개인별 time interval) |
| 피처 엔지니어링 | Two-stage reranker stage 2 (LightGBM + user/item/temporal features) — 시나리오 A Day 8-10 |
| 시간적 요소 | TiSASRec time embedding, Feb 27-29 spike 분석 ([eda_findings §5](eda_findings.md)) |

### 7.7 Two-stage Reranker — Feature Engineering 후보 (EDA 검증)

운영진 권고 ("효과적 파생변수 — 전환율 / 시간대 / category×brand / 선호도 / 가격대") 를 우리 EDA 결과로 평가 + 검증된 강한 시그널 기반 feature 후보 정리. 활용 시점은 시나리오 A Day 8-10 (Two-stage reranker) 시작.

#### 운영진 예시 vs 우리 EDA 강도

| 운영진 예시 | 우리 EDA 근거 | 강도 | 우리 처리 |
|---|---|:---:|---|
| 전환율 (view2cart / view2purchase / cart2purchase) | view2purch 0.07%, cart2purch 0.8% — 매우 sparse ([eda §6](eda_findings.md)) | ⚠️ | smoothed (Bayesian avg) 필수, **user-level X / item-level O** |
| 주요 구매 시간대 / 요일 | spike 빼면 약함 ([eda §15.5](eda_findings.md)) | ❌ | 후순위, cyclic encoding 가치 낮음 |
| Category + brand 상호작용 | IT brand 100% apparel anomaly — category-brand 거의 1:1 | ❌ | cross feature 새 정보 적음 |
| 선호도 (모호) | user-brand affinity 일치율 46.5% ([eda §15.4](eda_findings.md)) | ★★★ | brand affinity 는 강함, category 는 약함 |
| **주요 구매 가격대** | purchase 86.8% ∈ user view band ([eda §15.2](eda_findings.md)) | ★★★ | **★ 우리 데이터에서 가장 강한 시그널** |

→ 운영진이 가장 짧게 언급한 "가격대"가 우리 데이터에서 가장 강함. 시간/요일/카테고리 교차는 lift 기대 낮음.

#### Tier 0 — ⭐⭐⭐ Stage 1 모델 score (가장 강력)

Two-stage reranker 의 핵심 input. 다른 모든 feature 합친 것보다 강력. **stage 1 score 만 reranker 에 넣어도 single-model 보다 lift 큼.**

```
als_score(user, cand)        # exp_000 결과 활용
sasrec_score, tisasrec_score, mbstr_score, ease_score, ...
als_rank, sasrec_rank, ...   # 각 모델의 ranking 위치
score_sasrec_minus_als       # ensemble disagreement 시그널
```

#### Tier 1 — ★★★ User × Candidate Interaction (EDA 검증된 강한 시그널)

**A. User price band** — [eda §15.2](eda_findings.md) 86.8% coverage

```
cand_price_in_user_iqr        # binary: ∈ user view IQR (p25~p75)
cand_price_in_user_band       # binary: ∈ user_med ± 1.5*IQR
log_price_distance            # |log(cand_price / user_view_median)|
cand_price_zscore_to_user
```

**B. User-brand affinity** — [eda §15.4](eda_findings.md) 일치율 46.5%

```
cand_brand_is_user_top1_view  # binary: cand brand == user 최다 view brand
cand_brand_in_user_top3       # binary
user_brand_view_ratio         # user의 cand brand view 횟수 / total views
user_brand_view_rank          # 1, 2, 3, ..., outside_top5
```

**C. Recency / frequency on candidate** — [eda §15.1](eda_findings.md) 사전 view 이력 강력

```
user_view_count_of_cand           # raw + log1p
user_last_view_days_ago_cand      # recency
user_cart_count_of_cand           # 희귀하지만 강함 (cart→purch 10x view→purch)
cand_in_user_history              # binary
days_since_user_last_event        # user 자체 활성도
```

**D. Item repetition / NBR** — [eda §7](eda_findings.md) 반복도 14.6%

```
user_total_distinct_items_viewed
user_total_views                  # activity level
user_repeat_purchase_count
```

#### Tier 2 — ★★ Item-level + smoothed signals

**E. Smoothed conversion rates** — 운영진 예시 변형, item-level 한정

Bayesian smoothing 필수 — `(count + α * prior) / (denom + α)`, sparse 신호 보호:

```
cand_view_to_purchase_rate_smoothed
cand_view_to_cart_rate_smoothed
cand_cart_to_purchase_rate_smoothed   # cart 자체 sparse, 가장 위험
cand_global_popularity_rank
```

**F. Item-level static + G. User activity bucket**

```
cand_total_view_count, cand_brand_global_popularity
cand_price_relative_to_category       # 카테고리 평균 대비
user_total_events, user_event_recency_days
user_view_diversity, user_avg_session_length
```

#### Tier 3 — ☆ 후순위 (시간 남으면 ablation)

**H. 시간 패턴** ([eda §15.5](eda_findings.md) — 약한 시그널) / **I. Category cross** (single-domain 이슈)

```
cyclic_hour_sin/cos, user_purchase_hour_mode, user_active_dow_mode
cand_category_in_user_top3            # 우리는 거의 apparel 단일이라 효과 미미
```

#### 핵심 메시지

- **Tier 0 (Stage 1 model scores) > Tier 1-3 합** — reranker 첫 quick win 은 stage 1 score 통합
- **Tier 1 A (price band) + B (brand affinity)** 이 우리 EDA 에서 가장 강한 single signals
- 전환율은 sparse 문제 — smoothed item-level 만 신뢰
- 시간/요일/카테고리 교차는 lift 기대 낮음 → 시간 절약

---

## 8. 다음 실험 진입 시 결정 사항

각 새 실험 시작 전:

1. **목표 명확히**: candidate generation 다양성 확보 / 단독 score 푸시 / ensemble 멤버 추가 — 어떤 것?
2. **core/ 함수 활용**: 무조건 `core.data_loader`, `core.validation.time_based_split`, `core.metrics.ndcg_at_k_from_df`, `core.submission.predictions_to_submission` 사용 (재구현 X)
3. **predictions.parquet 표준 산출**: top-50 + score, 앙상블 입력 형식 유지
4. **자체 val로 ranking 비교, 절대값 신뢰 X**: Feb 27-29 spike 효과 인지
5. **wandb single run (train+inference 동일 id)**: exp_000에서 확립한 패턴
6. **README에 가설/하이퍼/결과 즉시 기록**
7. **제출은 진짜 쓸 만할 때만** (동점이면 제출 횟수 적은 쪽 우위 + calibration 차원에서 한 번은 필요)

---

## 9. 참고 — 우리가 이미 보유한 자산

| 자산 | 위치 | 용도 |
|---|---|---|
| ALS 베이스라인 self-val | exp_000 0.1838 | 모든 후속 실험의 ranking 기준점 |
| time_based_split | core/validation.py | val_days 파라미터 통일 |
| ID 매핑 캐시 | exp_000/saved/mappings/ | 동일 캐시 재사용 가능 (재학습 시 구조 동일하면) |
| EDA 인사이트 | docs/eda_findings.md | 모델 선택 정당화 |
| 베이스라인 SASRec 코드 | baseline/code/ (참조만, 복사 X) | RecBole 사용 패턴 참고 (yaml 구성 등) |
