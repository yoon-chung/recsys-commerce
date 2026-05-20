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

## 5. 우리 데이터 기준 우선순위 (next 3 후보)

EDA 권고 + ALS 결과 + 구현 비용 종합:

| 우선순위 | 후보 | 이유 |
|---|---|---|
| **1** | **TiSASRec** | EDA 1순위, RecBole에 즉시 있음, sequential 진입의 자연스러운 시작점. SASRec 베이스라인 코드 변형 가능 |
| **2** | **EASE (closed-form item-item)** | 학습 1분, ALS와 다른 family라 ensemble 다양성 ↑, 우리 sparse data에 강함. 구현 30줄. 의외의 강력 baseline |
| **3** | **BSARec 또는 GRU4Rec** | BSARec = SOTA 시도 (구현 비용 큼), GRU4Rec = RecBole 즉시 사용 + 우리 짧은 시퀀스(p90=29)에 적합 |

⚠️ **주의**: 자체 val 절대값을 절대 기준으로 삼지 말 것. exp_000이 self-val 0.18 (baseline 공시 public 0.08의 2.17배)을 찍은 건 Feb 27~29 spike 효과. 다음 모델들의 self-val을 ALS 0.18과 직접 비교하지 말고, 각 모델의 ranking + recall@10 등 보조 메트릭 + (가능하면) public 제출 점수로 종합 판단.

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
| **BSARec** | — | GitHub의 비공식 impl 포팅 | 가장 SOTA지만 구현 비용 |
| **TIFU-KNN** | — | 비공식 (논문 기준 구현) | NBR 보조 멤버 |
| Reranker (LightGBM) | lightgbm | feature engineering 필요 | candidate gen 후 stage 2 |

서버 미리 설치된 것: `recbole, kmeans_pytorch, ray, implicit, pyarrow, fastparquet, tqdm` (per CLAUDE.md).

---

## 7. 다음 실험 진입 시 결정 사항

각 새 실험 시작 전:

1. **목표 명확히**: candidate generation 다양성 확보 / 단독 score 푸시 / ensemble 멤버 추가 — 어떤 것?
2. **shared/ 함수 활용**: 무조건 `shared.data_loader`, `shared.validation.time_based_split`, `shared.metrics.ndcg_at_k_from_df`, `shared.submission.predictions_to_submission` 사용 (재구현 X)
3. **predictions.parquet 표준 산출**: top-50 + score, 앙상블 입력 형식 유지
4. **자체 val로 ranking 비교, 절대값 신뢰 X**: Feb 27-29 spike 효과 인지
5. **wandb single run (train+inference 동일 id)**: exp_000에서 확립한 패턴
6. **README에 가설/하이퍼/결과 즉시 기록**
7. **제출은 진짜 쓸 만할 때만** (동점이면 제출 횟수 적은 쪽 우위 + calibration 차원에서 한 번은 필요)

---

## 8. 참고 — 우리가 이미 보유한 자산

| 자산 | 위치 | 용도 |
|---|---|---|
| ALS 베이스라인 self-val | exp_000 0.1838 | 모든 후속 실험의 ranking 기준점 |
| time_based_split | shared/validation.py | val_days 파라미터 통일 |
| ID 매핑 캐시 | exp_000/saved/mappings/ | 동일 캐시 재사용 가능 (재학습 시 구조 동일하면) |
| EDA 인사이트 | docs/eda_findings.md | 모델 선택 정당화 |
| 베이스라인 SASRec 코드 | baseline/code/ (참조만, 복사 X) | RecBole 사용 패턴 참고 (yaml 구성 등) |
