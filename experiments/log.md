# Experiments Log

모든 실험의 단일 lab notebook. 각 실험 폴더에는 코드+config만 두고, **가설/결과/학습은 이 파일에 작성**.

**대회**: Commerce Behavior Purchase Prediction · **Train**: 2019-11-01 ~ 2020-02-29 (8.35M events, 638,257 users × 29,502 items) · **평가**: 2020-03-01 ~ 2020-03-07, NDCG@10 binary, public/private 50:50
**제출 규정**: 동점 시 제출 횟수 적은 쪽 우위 → **무의미한 제출 회피**

**자체 val 규약**: train 마지막 7일 (Feb 23-29) hold-out + `restrict_to_train=True` + `gt_event_types=['purchase']` + `eval_users` 928명. exp_001 의 `val_gt.parquet` + `eval_users.json` 을 후속 실험 모두 재활용.

**Calibration** (family 별):
- ALS / EASE: ÷ 2.32 (exp_000 측정)
- BSARec (sequential): ÷ 2.53 (exp_002/002b/002e 3회 검증, 오차 ±0.0005)
- TIFU-KNN (classical KNN): ÷ 2.49 (exp_007 측정)
- val 의 99.7% 가 Feb 27-29 spike → self-val 은 사실상 spike 예측 skill만 측정.

---

## 베이스라인 + leaderboard (live)

| Exp | Model | Family | Val NDCG@10 | Val recall@10 | Public | Status |
|---|---|---|---:|---:|---:|---|
| (baseline) | ALS (주최사) | MF | — | — | 0.0847 | 참조 |
| (baseline) | SASRec (주최사) | sequential | — | — | 0.0842 | 참조 |
| exp_000 | ALS | MF | 0.1838 | 0.2558 | 0.0791 | DONE (calibration) |
| exp_001 | EASE | item-item | 0.1848 | 0.2909 | — | DONE |
| ensemble_v1 | ALS+EASE RRF | fusion | 0.1725 | 0.2624 | — | negative |
| exp_002 | BSARec (4m) | sequential | 0.2391 | 0.3195 | 0.0943 | DONE |
| ensemble_v2 (equal) | ALS+EASE+BSARec RRF | fusion | 0.2067 | 0.3207 | — | negative −0.032 |
| ensemble_v2 (1:1:2) | weighted RRF | fusion | 0.2031 | 0.3068 | — | negative −0.036 |
| exp_002b | BSARec (4w_holdout) | sequential | 0.2414 | 0.3242 | 0.0955 | DONE |
| exp_002c | BSARec (2w_holdout) | sequential | 0.2374 | 0.3240 | — | DONE (−0.0040 vs 4w) |
| exp_002d | BSARec (1w_holdout) | sequential | 0.2395 | 0.3168 | — | DONE (−0.0019 vs 4w) |
| exp_002e | BSARec (4w_full, spike+) | sequential | 0.2470\* | 0.3274\* | 0.0975 | DONE |
| exp_002g | BSARec (2w_full, spike+) | sequential | 0.2479\* | 0.3304\* | 0.0975 | DONE (tie with 002e) |
| exp_002f | BSARec (1w_full, spike만) | sequential | 0.2408\* | 0.3190\* | — | DONE (less data hurts) |
| exp_003 | DiffRec | diffusion | 0.1543 | 0.2257 | — | DONE (< ALS, paradigm coverage 만) |
| exp_004 | FEARec | sequential FFT+autocorr | — | — | — | FAILED (RecBole 1.2 hang) |
| exp_005 | BERT4Rec | bidirectional MLM | 0.2158 | 0.3061 | — | DONE (killed ep13, 추정 public ~0.085 → 제출 X) |
| exp_006 | BSARec+CL hybrid (4w) | sequential + contrastive | 0.2347 | 0.3188 | — | DONE — LOO 0.2646/0.4441 했지만 our val 에선 BSARec 보다 ↓ (제출 X) |
| exp_007 | TIFU-KNN (4m) | classical KNN + temporal freq | 0.2922 | 0.3915 | 0.1175 | DONE — single-best transformer kill |
| exp_007b | TIFU-KNN multi-behavior (view=1,cart=3,purchase=5) | classical KNN | 0.2933 | 0.3915 | — | DONE — self-val +0.0011 vs exp_007 (제출 X, ensemble lift X) |
| exp_009 | MB-STR v1 (BSARec + behavior emb, hidden=64) | multi-behavior seq | 0.2389 | 0.3221 | — | DONE — 제출 X (단독 TIFU 보다 낮음) |
| exp_009b | MB-STR v2 / v2b (hidden=256) | multi-behavior seq | — | — | — | ABORTED — v2 plateau, capacity scaling 으로는 v1 LOO 0.2646 못 넘음 |
| ensemble_v3-v7 | 5 variants (RRF / weighted / z-score / MB base / clean 3-model) | fusion | best 0.293 (=TIFU only) | — | — | ALL DEAD-END — simple aggregation 한계, 상세는 §Ensemble v3-v7 |
| exp_010 | LGBM reranker binary (5-fold on 928 eval users) | 2-stage rerank | 0.3418 (OOF) | 0.4258 | 0.1353 | DONE — +0.0178 vs TIFU (+15.2%) |
| **exp_010b** | **LGBM reranker LambdaRank (objective swap)** | **2-stage rerank** | **0.3513 (OOF)** | **0.4369** | **0.1358** | **DONE — +0.0095 OOF vs binary, public lift 작음 (calibration ratio 차이)** |
| (미실행) | exp_008 LLM-as-Reranker on TIFU top-50 | LLM rerank | — | — | — | 미실행 — Week 2 LLM advisor (service/) 로 흡수 |

\* full-data 변종은 val 포함 학습 → self-val 인플레이션. 의사결정은 public 기준.

---

## 제출 이력

| # | Date | Exp | Public | 핵심 학습 |
|---:|---|---|---:|---|
| 1 | 2026-05-20 | exp_000 ALS | 0.0791 | 베이스라인 공시 0.0847 대비 −6.6% gap ≈ 7/120일 holdout 비율. 파이프라인 정상 |
| 2 | 2026-05-21 | exp_002 BSARec | 0.0943 | +11.3% vs ALS 베이스라인. sequential calibration ratio 2.535 측정 |
| 3 | 2026-05-21 | exp_002b BSARec 4w | 0.0955 | calibration framework 정확도 검증: 예측 0.0953 vs 실제 0.0955 (오차 0.0002) |
| 4 | 2026-05-22 | exp_002e BSARec 4w_full | 0.0975 | spike 포함 retrain 효과. 예측 0.0977 vs 실제 0.0975 |
| 5 | 2026-05-22 | exp_002g BSARec 2w_full | 0.0975 | 002e 와 tie. tie-break 상 002e 가 002g 보다 상위 (제출 횟수) |
| 6 | 2026-05-22 | exp_007 TIFU-KNN 4m holdout | **0.1175** | **+20.5% vs BSARec, new leaderboard top**. TIFU-KNN family calibration ratio **2.487** 측정 (BSARec 2.53 / ALS 2.32 사이). **Classical KNN + temporal frequency 가 transformer 4종 (BSARec, BERT4Rec) 를 압도** — repeat-purchase + spike 가 dominant signal 이었음 |
| 7 | 2026-05-26 | exp_010 LGBM reranker binary (5 stage-1 models, 51 features) | **0.1353** | **+0.0178 vs TIFU (+15.2%), Reranker effectiveness 입증**. OOF NDCG 0.342 / public 0.1353 → calibration ratio **2.527** (TIFU 2.49 / BSARec 2.53 사이, family-agnostic framework 신뢰 강화). Top features: tifu_rank > ui_days_since_last > tifu_norm_score > item_cart > user_repeat_ratio — TIFU backbone + popularity/recency 보조 패턴. **LGBM 은 LOO-overfit 모델 (BERT4Rec, BSARec_CL) 도 자동 down-weight 하여 활용** (RRF 가 못 했던 일) |
| 8 | 2026-05-27 | exp_010b LGBM reranker LambdaRank (objective swap) | **0.1358** | OOF +0.0095 (0.3418 → 0.3513) 명확한 lift 인데 **public 은 +0.0005 만 전이**. Calibration ratio 2.527 → 2.587 변화 — **LambdaRank 의 per-user objective 가 small eval (928 user) 에 over-fit 더 큼**. Empirical lesson: "OOF improvement ≠ public improvement". Self-val sample 작을 때 ranking objective 의 per-group fit 위험. |

---

## 제출 전 체크리스트

`core.submission.validate_submission()` 자동 검증 + 사람이 한 번 더:

- [ ] `output.csv` (6,382,570 × 2), 638,257 user × 10 item, user 내 중복 0
- [ ] **self-val 이 베이스라인 (÷2.53 → 0.0847) 라인 명백히 초과** 또는 ensemble 다양성 목적 명확
- [ ] `predictions.parquet` (top-50 + score) 함께 저장 → ensemble 입력 가능
- [ ] wandb artifact 업로드, leaderboard + 제출 이력 갱신
- [ ] `git status` — 데이터/베이스라인/raw ID 파일 staging X

---

# 실험 기록 (요약)

각 실험: 가설 → 결과 → 핵심 학습. 상세 하이퍼는 해당 폴더 `config.yaml`, 알고리즘 설명은 [docs/references.md](../docs/references.md).

---

## exp_000_als_baseline — DONE (2026-05-20)

[code](./exp_000_als_baseline/)

**목적**: 점수가 아니라 (1) reference (2) `core/` 인프라 검증 (3) self-val ↔ public calibration 측정.

**구현**: `implicit.als.AlternatingLeastSquares` (GPU). 베이스라인 하이퍼 매칭 (`factors=32, reg=0.001, alpha=10, filter_already_liked=false, event_weights=1/1/1`). 베이스라인 코드 참조 금지, implicit 컨벤션 따라 직접 구현.

**결과**: self-val 0.1838 / public 0.0791. **공시 0.0847 대비 −6.6% gap ≈ 7/120일 holdout 비율** — 파이프라인 정상.

**핵심 학습**:
1. **Calibration ratio 2.32 확정** (ALS 기준). sequential 모델은 별도 측정 필요
2. **self-val 절대값 inflate** — Feb 27-29 spike. 같은 split 내 상대 비교만 신뢰
3. **1차 mismatched config (factors=128 / reg=0.01 / alpha=40 / weights 1·3·5 / filter_already_liked=true)**: self-val 0.0288 — `filter_already_liked=false` 가 spike 의 "직전 view → 구매" 패턴 차단 안 한 게 6.4× lever

---

## exp_001_ease — DONE (2026-05-21)

[code](./exp_001_ease/) · **closed-form** `B = inv(X^T X + λI)` row-normalized, 30줄

**가설**: item 반복도 14.6% + view-dominant 99.78% → item-item similarity 강력 시그널. ALS (MF) vs EASE (item-item) family-diverse RRF lift 기대.

**결과**: self-val 0.1848 (vs ALS 0.1838, +0.5%) / recall 0.2909 (+13.7% vs ALS). 학습 5분 (Cholesky), inference 9분.

**핵심 학습**: NDCG 비등 + recall 큰 차이 → EASE 는 정답을 top-10 에 **더 자주** 잡지만 순위는 ALS 와 비슷. 제출 안 함 (calibration ratio 적용 시 ALS public 과 동일 예상).

---

## ensemble_v1 (ALS+EASE) — DONE 2026-05-21 (negative)

[code](./ensemble_v1_als_ease/) · RRF `1/(k+rank)`, k=60

**결과**: NDCG 0.1725 — ALS 대비 **−0.011**, EASE 대비 **−0.012**. recall vs EASE −0.028.

**핵심 학습**: 둘 다 implicit-feedback CF → **family 동질성**. 다른 알고리즘 ≠ 다른 시그널. EASE 가 ALS dominate (recall +13.7%) → 약한 ALS 시그널 동등 가중이 noise 추가. **family-diverse 가 진짜 변수** 가설 → exp_002 BSARec 이후 재검증.

---

## exp_002_bsarec — DONE (2026-05-21)

[code](./exp_002_bsarec/) · **AAAI 2024**. SASRec backbone 의 TransformerEncoder 자리에 `BSARecEncoder` 삽입. 각 layer: `α·FrequencyLayer(x) + (1-α)·MHA(x)`. FrequencyLayer = `low_pass + sqrt_beta² · high_pass` after rFFT/irFFT. [fra.py](./exp_002_bsarec/fra.py) 에 저자 코드 (Apache-2.0) 포팅.

**하이퍼**: alpha=0.7, c=5 (Beauty default), n_layers=2, hidden=64, batch=2048, lr=0.002. epoch 52 best (LOO 0.2438), ~3h.

**결과**: self-val 0.2391 / recall 0.3195 / public **0.0943** (vs ALS 베이스라인 +11.3%). RecBole LOO 0.2438 ↔ self-val 0.2391 거의 일치 — self-val 신뢰성 확인.

**핵심 학습**:
1. **Sequential family 가 진짜로 다른 시그널** — ALS/EASE (0.185 라인) 대비 +30%
2. **Sequential calibration ratio 2.535** (ALS 의 2.32 보다 +9% 큼) — sequential self-val 이 더 inflate
3. ensemble_v1 의 "family diversity 가설" 데이터 검증 → ensemble_v2 (3-family) 진행

---

## ensemble_v2 (ALS+EASE+BSARec) — DONE 2026-05-21 (negative)

[code](./ensemble_v2_als_ease_bsarec/) · equal + weighted (BSARec 2x) 둘 다 시도

**결과**:
| 시도 | NDCG | Δ vs BSARec |
|---|---:|---:|
| equal 1:1:1 | 0.2067 | −0.0324 |
| weighted 1:1:2 | 0.2031 | −0.0360 (**더 나쁨**) |

**핵심 학습** — 3번 다른 가설 모두 negative 구조 진단:
1. v1 negative = **family 동질성** (CF only)
2. v2 equal negative = **strength 불균형** (BSARec +30%, ALS/EASE 가 top rank 희석)
3. v2 weighted negative = **BSARec false positive 도 amplify** (recall −0.013 이 증거). ALS/EASE 의 견제 약화

**핵심 결론**: 단순 negative 가 아니라 structural limitation 을 ablation 으로 진단. ensemble 접근 dead (BSARec → ∞ 가 BSARec 단독에 수렴).

---

## exp_002b/c/d (BSARec recency ablation, holdout) — DONE 2026-05-21

Window 별 ablation. 4m baseline → 4w/2w/1w 변종. 모델/하이퍼 동일, 학습 데이터만 변경.

| 변종 | self-val NDCG | recall | Public | Δ vs 4m public |
|---|---:|---:|---:|---:|
| 4m (exp_002) | 0.2391 | 0.3195 | 0.0943 | — |
| **4w (exp_002b)** | **0.2414** | **0.3242** | **0.0955** | **+0.0012** |
| 2w (exp_002c) | 0.2374 | 0.3240 | — | — |
| 1w (exp_002d) | 0.2395 | 0.3168 | — | — |

**핵심 학습**:
1. **4w recency 효과 진짜였음** (+0.0012 public) — 작지만 노이즈 아님. calibration 모델 예측 (0.0953) 과 실제 (0.0955) 오차 0.0002
2. **Calibration ratio 2.53 안정** (4m 2.535 / 4w 2.528) — recency 가 ratio 자체는 안 바꿈
3. **Sub-val 진단 불가**: val 1,223 purchase 중 **Feb 23-26 (no_spike) 단 4건 (0.3%)**, Feb 27-29 spike 99.7%. `full ≈ spike_only`. Distribution shift 직접 진단 불가능한 데이터
4. 2w 가 4w 보다 나쁨 (−0.0040) → 학습 데이터 줄이는 게 능사 아님. **4w 가 sweet spot**

---

## exp_002e/f/g (BSARec full-data, val 포함 retrain) — DONE 2026-05-22

**가설**: 4w holdout 0.0955 가 final-submission style (val 포함 학습) 로 추가 lift. spike 포함 효과 측정.

| 변종 | Self-val\* | recall\* | Public |
|---|---:|---:|---:|
| **4w_full (exp_002e)** | **0.2470** | **0.3274** | **0.0975** |
| 2w_full (exp_002g) | 0.2479 | 0.3304 | 0.0975 (tie) |
| 1w_full (exp_002f) | 0.2408 | 0.3190 | — |

\* val 포함 학습으로 self-val 인플레이션 포함.

**핵심 학습**:
1. **4w_full retrain 효과 큼** (Δ vs 4w_holdout: public +0.0020). Final-submission style 작동
2. **Calibration ratio 2.533** (002b 2.528 와 사실상 동일 — 3번째 검증). 예측 0.0977 vs 실제 0.0975
3. 2w_full self-val 이 4w_full 보다 살짝 높지만 (+0.0009) public 은 tie → 추가 self-val gain 은 **spike 외우기** 비중 컸음, Mar 1-7 transfer 안 됨
4. 1w_full 은 spike-only 학습 효과 — less data hurts (−0.0062 vs 4w_full self-val)
5. **Top: exp_002e (0.0975)**. tie-break 상 #4 가 #5 보다 상위

---

## exp_003_diffrec — DONE 2026-05-22 (paradigm coverage, < ALS)

[code](./exp_003_diffrec/) · **SIGIR 2023** generative. user binary interaction vector (29.5k) → Gaussian noise → DNN denoise → reconstructed score.

**결과**: self-val 0.1543 / recall 0.2257 — **ALS (0.1838) 보다 낮음**. 44% cold-start (user_inter_num_interval filter side-effect로 활성 user 만 학습).

**핵심 학습**: paradigm 다양성 차원으로만 보존. submission 안 함 (베이스라인 미달). Generative 가 "user intent distribution" 학습한다는 가설은 99.78% view-dominant + spike-dominant 데이터에서 성립 안 함.

---

## exp_004_fearec — FAILED 2026-05-22

[code](./exp_004_fearec/) · **SIGIR 2023** FFT + autocorrelation.

**문제**: RecBole 1.2 에서 `trainer.fit()` 초기화 hang (CPU 99%, 30+분 stuck, GPU 0%). `--no-wandb` 도 동일. import 자체는 통과.

**대응**: 포기, BERT4Rec 으로 대체. 한 줄 결론: "RecBole 1.2 FEARec incompatibility at scale".

---

## exp_005_bert4rec — DONE 2026-05-22 (Day 1B, killed ep13, 제출 X)

[code](./exp_005_bert4rec/) · **CIKM 2019** bidirectional MLM Cloze.

**가설**: BSARec/SASRec 의 causal next-item 과 다른 학습 신호 (bidirectional context). Sequential family 내 second signal.

**하이퍼**: n_layers=2, hidden=64, n_heads=2, mask_ratio=0.2, max_seq=50, batch=2048, lr=0.001. eval_step=2.

**학습 trajectory** (RecBole LOO):
| Epoch | NDCG@10 | recall@10 | Δ NDCG (2-ep) |
|---:|---:|---:|---:|
| 1 | 0.1762 | 0.3087 | — |
| 3 | 0.1885 | 0.3275 | +0.0123 |
| 5 | 0.1940 | 0.3369 | +0.0055 |
| 7 | 0.1980 | 0.3435 | +0.0040 |
| 9 | 0.1979 | 0.3431 | −0.0001 |
| 11 | 0.1999 | 0.3479 | +0.0020 |
| **13 (best)** | **0.2006** | **0.3485** | +0.0007 |

상승률 반감 페이스 → ep13 plateau 임박. TIFU-KNN 0.1175 결과 후 GPU 회수 우선 → manual kill.

**우리 self-val 결과** (vs RecBole LOO 갭이 핵심 발견):

| 메트릭 | RecBole LOO | our 7-day val | Δ |
|---|---:|---:|---:|
| NDCG@10 | 0.2006 | 0.2158 | +0.0152 |
| recall@10 | **0.3485** | **0.3061** | **−0.0424** |

**핵심 발견 — bidirectional MLM 의 task mismatch**:

비교: BSARec 패턴에서는 RecBole LOO 가 self-val 보다 *높음* (transformer causal 의 정상 패턴). BERT4Rec 는 **반대 방향**:
- LOO = "마지막 item 마스킹 → 예측" → Cloze training 과 **정확히 일치** → over-specialized
- 우리 val = "7일 윈도우 multi-purchase 예측" → next-item 가정 깨짐
- **Bidirectional MLM 이 multi-target purchase task 에 transfer 약함**
- 핵심 결론: "**모델 training task 와 inference task 정합성**이 RecSys 특유의 함정 — BERT4Rec 이 LOO 에 over-fit 하는 사례를 데이터로 잡음"

**추정 public**: 0.2158 / 2.53 ≈ **0.0853** (ALS 베이스라인 수준) → **제출 X**.

**Ensemble 가치 재평가**:
- recall 0.3061 ≈ BSARec recall 0.3195 → 이전 가설 "다른 후보 set" 약화
- TIFU recall 0.3915 와 큰 격차 → ensemble 시 BSARec 1개로 충분, BERT4Rec 추가 lift 작을 듯
- 결정: ensemble v3 후보로 보관, but 핵심 아님

---

## exp_007_tifu_knn — DONE 2026-05-22 (Day 1B+, NEW LEADERBOARD TOP)

[code](./exp_007_tifu_knn/) · **SIGIR 2020** "Modeling Personalized Item Frequency for Next-Basket Recommendation" (He et al.). Non-neural classical method.

**왜 이걸 추가**:
- 우리 5개 모델 (ALS/EASE/BSARec/DiffRec/BERT4Rec) 다 scoring 기반 deep/closed-form → **KNN/frequency-based paradigm 0개**. 빈 슬롯 채우기
- Pure transformer pipeline 의 한계 가설: repeat-purchase + temporal frequency 가 retail data 의 dominant signal 일 가능성 (EDA item 반복도 14.6% + Feb 27-29 spike 가 시사)
- Production e-commerce 의 first-class signal: "user X 가 item Y 를 반복 view/cart" 패턴. transformer max_seq=50 cap 보다 직접 모델링
- **핵심 결론**: "Classical method 가 같은 데이터에서 deep model 을 이긴 이유 — repeat-purchase + temporal frequency 가 dominant signal" — research → production maturity 신호

**알고리즘**:
```
1. user history → G=7 chronological groups
2. user_vec[u, i] = Σ_g (r_a^(G-1-g)) · Σ_t∈g (r_w^(L_g-1-pos)) · event_w
3. row-L2-normalize → cosine sim KNN (top-K=300 neighbors)
4. score(u, i) = α · user_vec[u, i] + (1-α) · mean(neighbors)
```

**하이퍼** (paper default): group_count=7, decay_within=0.9, decay_across=0.7, knn_k=300, alpha=0.7, event_weights=1/1/1.

**Window 선택 — 4m**:
- TIFU 의 decay (`r_a=0.7`) 가 **이미 recency 처리** → 외부 4w cut 은 double-cut + 정보 손실
- `group_count=7` × 4m = bucket 당 ~17일 (spike 한 bucket 에 적정 비중). 4w 면 bucket ~4일, spike 1 bucket dominant
- KNN 이웃 다양성 → 638k user pool 클수록 sim 매칭 정확
- Multi-behavior rare event (cart 660k, purchase 70k) 절대 빈도 보존 필요

**구현**:
- numpy/scipy 만 (GPU 불필요) → BERT4Rec GPU 점유 중 CPU 로 병행 가능
- `tifu_knn.py`: TIFUKNN 클래스, 100줄. sparse CSR user × item matrix
- KNN: 1024 batch × 638k users 코사인 sim → top-K. 예상 ~30-60분 (server 64-core)
- 학습 (matrix build): ~5분. inference (KNN+score): ~60분 추정

**결과**:

| 메트릭 | TIFU-KNN | vs exp_002e BSARec (이전 top) |
|---|---:|---:|
| self-val NDCG@10 | **0.2922** | +0.0452 (**+18.3%**) |
| self-val recall@10 | **0.3915** | +0.0641 (**+19.6%**) |
| **Public NDCG@10** | **0.1175** | **+0.0200 (+20.5%)** |
| Calibration ratio | **2.487** | BSARec 2.53 / ALS 2.32 사이 |
| 학습 시간 | ~5분 (matrix build) | BSARec ~3h, ratio 차이 작음에도 짧음 |
| Inference 시간 | ~38분 (KNN compute, 64-core CPU) | BSARec ~5분 (GPU) |

**핵심 학습**:
1. **Classical KNN + temporal frequency 가 transformer 4종을 18-20% 차이로 압도**
   - 같은 데이터에서 BSARec 0.247, BERT4Rec 0.20 < TIFU 0.292
   - Pure transformer pipeline 이 retail data 에 최적 아님 — paper 베끼기 paper-chasing 의 위험성
2. **Mechanism 진단**:
   - TIFU 의 user_vec[u, i] = 가중 freq 가 **"user X 가 item Y 를 반복 view/cart"** 패턴 직접 모델링
   - BSARec/BERT4Rec 의 max_seq=50 + parametric attention 은 이 signal 을 압축/추상화하며 손실
   - Feb 27-29 spike → temporal decay (`r_a=0.7^g`) 가 spike 가까운 group 자동 강조 → exactly the val distribution
3. **Calibration ratio 안정성 확장**: 3개 family (ALS 2.32 / BSARec 2.53 / TIFU 2.49) 모두 좁은 범위 내. **Family 무관하게 calibration framework 작동** — 미래 모델 의사결정에 신뢰 도구
4. **Production e-commerce 연결**: 쿠팡/Amazon 의 multi-stage retrieval 에서 classical KNN/co-visit 이 first-stage candidate generator 로 살아있는 이유 — repeat-frequency 가 retail 의 first-class signal
5. **Recall 0.3915 >> BSARec 0.3274** → ensemble 시 매우 다른 후보 set 제공. Reranker 가치 ↑

**한 줄 요약**:
> "Sequential transformer 4종 (BSARec / BERT4Rec / BSARec+CL hybrid) 를 ablate 했지만, 같은 데이터에서 100줄짜리 2020 paper KNN method (TIFU-KNN) 가 **public NDCG +20.5%** 로 압도했다. 진단해 보니 이 데이터의 dominant signal 인 **user-item repeat frequency + temporal decay** 를 transformer 의 parametric attention 으론 직접 잡지 못했기 때문이다. 이게 production e-commerce 에서 classical KNN/co-visit 류가 first-stage candidate generator 로 살아남는 mechanism. Research SOTA 와 production reality 사이의 gap 을 데이터로 직접 검증."

**후속 작업**: TIFU 위에 multi-behavior weight ablation (exp_007b), 5종 ensemble (v3-v7), 2-stage LGBM reranker (exp_010/010b) 진행. 결과적으로 **LGBM reranker 가 best lever** (single TIFU +15.2% public lift).

---

## exp_006_bsarec_cl — DONE 2026-05-22 (Day 1C, 제출 X)

[code](./exp_006_bsarec_cl/) · BSARec backbone + CL4SRec InfoNCE 보조 loss (novel combination — 두 paper 첫 결합).

**구현**: `BSARecCL(BSARec)` 클래스. `calculate_loss() = rec_loss + lmd · InfoNCE(z1, z2)`. z1, z2 = forward(aug1), forward(aug2). aug = crop / mask / reorder 중 random. RecBole left-padded 레이아웃에 맞춰 augmentation 재작성.

**하이퍼**: BSARec (alpha=0.7, c=5) + CL4SRec (lmd=0.1, tau=1.0, crop=0.4, mask=0.3, reorder=0.4) 모두 paper default.

**학습**: 4w_holdout (cy_commerce_4w 데이터셋, exp_002b 와 동일). epoch 51 best @ LOO 0.2646, 211분 (3.5h).

**결과**:

| 메트릭 | RecBole LOO | our 7-day val | Δ |
|---|---:|---:|---:|
| NDCG@10 | 0.2646 | **0.2347** | −0.0299 |
| recall@10 | 0.4441 | **0.3188** | **−0.1253** |

**예상치 못한 발견** — LOO 강력하지만 our task transfer 약함:
- BSARec 4w_holdout (exp_002b): LOO 0.2658 → our val 0.2414 (Δ −0.024)
- **BSARec+CL hybrid**: LOO 0.2646 → our val 0.2347 (Δ −0.030) — gap 더 큼
- LOO recall 0.4441 (BSARec 보다 +0.10) 의 시그널이 our val 에는 transfer 안 됨

**Mechanism**: BERT4Rec (§exp_005) 와 같은 패턴 — CL4SRec contrastive aug 의 학습 신호가 LOO last-item-masking 과 구조적으로 유사 → LOO 메트릭에 over-specialized, multi-target 7-day val 에는 transfer 약함. **Self-supervised/contrastive 가 LOO 에 over-fit 하는 함정의 2번째 사례** (종합 패턴 → §LOO over-specialization 2건 발견).

**Cold-start 38,523명** (BSARec 4w_holdout 의 14k 대비 2.7배 많음):
- 같은 cy_commerce_4w 데이터셋이지만 RecBole 의 user_inter_num_interval 필터가 BSARecCL 학습 시 다르게 작동한 듯
- popularity fallback 비중 ↑ → public 점수에 negative 영향

**예상 public**: 0.2347 / 2.53 ≈ **0.093** (BSARec 4w_holdout 0.0955 보다 ↓, BSARec 4w_full 0.0975 보다 ↓, TIFU 0.1175 보다 한참 ↓) → **제출 X**.

**Ensemble 가치**: LOO 기반 recall 0.44 시그널이 our val 에서 0.32 로 떨어진 만큼 ensemble 가치도 작아짐 (BSARec recall 0.33 과 거의 동일). 후보 set 다양성으로 보면 약함.

---

## exp_007 추가 분석 — TIFU vs BSARec 진단 노트북

별도 deliverable: [docs/diagnosis_tifu_vs_bsarec.ipynb](../docs/diagnosis_tifu_vs_bsarec.ipynb) · 24 cells (15 markdown + 9 code), pre-filled outputs.

5개 segment-level 분석으로 "classical 이 deep 을 이긴 이유" 정량 증명:
- **A2 repeat-buyer (56% of eval)**: TIFU +0.053 NDCG — frequency 가설 확정
- **A3 long-history user (21%)**: TIFU +0.091 NDCG — BSARec max_seq=50 cap 정보 손실
- **A4 top-100 popular items (34% of GT)**: TIFU +9.2% hit rate
- **A1 prediction overlap mean 36%** — 64% diverge, ensemble 가치
- **A5 per-user**: TIFU win 19.3% vs BSARec win 11.5% (1.68배)

전체 분석은 ipynb 참고.

---

# 2026-05-22 — TIFU-KNN 충격 후 전략 pivot

exp_007 (TIFU-KNN) 가 BSARec 4종을 +18-20% 차이로 압도. Transformer 추가 lift 추구의 ROI 가 낮음을 정량 확인 → **TIFU-KNN 위에 ensemble + LGBM reranker** 로 score-lift lever 전환. Transformer 계열 (exp_006 BSARec+CL, MB-STR) 은 paradigm coverage 용 ablation 으로만 진행 (점수 목적 X).

**Window 선택 원리** (model mechanism 별):
- Sequential causal (BSARec, SASRec): recency 유리 → 4w
- Self-supervised / generative (BERT4Rec MLM, DiffRec): sample 다양성 유리 → 4m
- Multi-behavior / rare event (MB-STR): 절대 빈도 유리 → 4m
- Reranker (LLM, LGBM): candidate generator window 따라감

새 모델마다 4종 window 다 돌리지 않음. mechanism 기반 1개만 선택, winner 만 val-포함 retrain.

---

**LOO over-specialization 2건 발견**:

| Model | RecBole LOO NDCG | our 7-day val NDCG | Δ |
|---|---:|---:|---:|
| BSARec (causal) | 0.2658 | 0.2414 | −0.0244 (정상) |
| **BERT4Rec (bidirectional MLM)** | **0.2006** | **0.2158** | **+0.0152 (반대 방향)** |
| **BSARec+CL hybrid (contrastive)** | **0.2646** | **0.2347** | **−0.0299 (gap 더 큼)** |
| TIFU-KNN | — (LOO 미사용) | 0.2922 | — |

**Pattern**: Self-supervised (BERT4Rec Cloze) + contrastive (BSARec+CL InfoNCE) 가 RecBole LOO 의 last-item-masking task 에 over-specialized → multi-target 7-day val 에는 transfer 약함. **RecBole default metric 이 우리 task 와 misalign 될 수 있다는 사례 2건 확보**.

---

## exp_007b_tifu_mb — DONE 2026-05-26 (제출 X)

[code](./exp_007b_tifu_mb/) · exp_007 의 multi-behavior weight ablation. event_weights view=1 / cart=3 / purchase=5.

**가설**: TIFU 1/1/1 baseline 의 multi-behavior weight ablation — cart/purchase 같은 conversion signal 이 view 보다 강한 lever 인지 직접 검증.

**결과**: self-val NDCG **0.2933** (exp_007 0.2922 대비 +0.0011, 거의 동일). 예상 public ≈ 0.118 (lift 미미). **단독 제출 X**.

**Ensemble 효과 (v6 에서 확인)**: TIFU(MB) 를 base 로 한 ensemble 조차 negative — multi-behavior weighting 자체로는 ensemble lift 안 살아남. Lift 가 살아나려면 reranker 필요 (exp_010 LGBM 으로 입증).

---

## exp_009_mbstr_v1 — DONE 2026-05-25 (제출 X)

[code](./exp_009_mbstr/) · BSARec backbone + behavior embedding (간소화 MB-STR). `e_i = item_emb + behavior_emb + position_emb`.

**구현**: paper full 의 behavior-aware FFN / cross-behavior attention 은 skip (시간 제약). BSARec FFT 그대로 + 3 behavior embedding (view=1/cart=2/purchase=3, PAD=0) 만 추가.

**하이퍼**: hidden=64, n_layers=2, n_heads=2, inner=256 (BSARec exp_002 와 동일).

**결과**: self-val NDCG **0.2389** / recall **0.3221** / LOO NDCG **0.2646**. TIFU 0.2922 보다 낮음. 예상 public ≈ 0.094 → **단독 제출 X**.

**Lesson — capacity gap**: paper-faithful MB-STR config 는 hidden=256, n_layers=3, n_heads=4, inner=512 (우리 v1 의 **16배 capacity**). v1 underfit 가설 → v2/v2b 시도 (아래).

---

## exp_009b_mbstr_v2 / v2b — ABORTED 2026-05-26

[code](./exp_009_mbstr/) · `config_v2.yaml`, `config_v2b.yaml`. capacity 4-16x 증가 시 self-val 도 올라가는지 검증.

### v2 — ABORTED (epoch 31, valid 0.2538 < v1 0.2646)

하이퍼: hidden=256, n_layers=3, n_heads=4, inner=512, batch=4096, lr=0.002, dropout=0.5/0.5.

**문제**: 5.5h 학습 후 valid_score plateau ≈ 0.252-0.254. v1 LOO 0.2646 못 따라잡음. train loss 도 매우 느린 감소 (7894→7870 over 60min) → local optimum 탈출 실패. **Lower LR + lower dropout 필요** 진단.

### v2b — ABORTED

조정: learning_rate 0.002 → 0.001, dropout 0.5 → 0.3 (보수안). 다른 하이퍼 v2 와 동일.

**결과**: v2 와 유사한 plateau 패턴 확인 + Week 1 종료 시점 도달 → archive. **v1 (hidden=64) 가 우리 eval 환경 (928 user / 1,443 purchase) 의 capacity sweet spot** 으로 잠정 결론. MB-STR 트랙 종결.

---

## Ensemble v3-v7 — ALL DEAD-END (2026-05-23~26)

5 stage-1 모델 (TIFU/BSARec/MB-STR/BSARec+CL/BERT4Rec) 의 모든 score-blind / weighted ensemble.

| Version | Method | 최선 self-val | Δ vs single TIFU 0.2933 |
|---|---|---:|---:|
| v3 | RRF (rank-based) + grid sweep | ~0.282 | −0.011 (best 도 negative) |
| v4 | per-user min-max + weighted score | — | negative (all configs) |
| v5 | per-model z-score | — | negative |
| v6 | exp_007b TIFU(MB) base + 동일 stack | — | negative |
| v7 | **clean 3-model** (TIFU + BSARec + MB-STR, LOO-overfit 2개 제거) | 0.293 (=TIFU only) | best `t5_b1_m1` 도 −0.007 |

### 가설 H 기각 (v7 의 의미)

**Hypothesis H**: v3-v6 dead-end 의 원인이 BSARec+CL + BERT4Rec (LOO-overfit) 의 ensemble 오염일 것.  
**Test**: v7 = 같은 stack 에서 2개 LOO-overfit 제거 → clean 3-model RRF.  
**Result**: 여전히 negative. weight 가 TIFU 쪽으로 갈수록 단조 회복 → asymptote = single TIFU.  
**결론**: pollution 이 원인 아님. 우리 환경에서 **simple RRF 가 본질적으로 작동 안 함**. 모델 quality gap (TIFU 0.293 vs BSARec 0.247 vs MB-STR 0.239) 때문에 RRF aggregation 이 단일 winner 못 넘음. Score-blind aggregation 의 구조적 한계 — **모델 quality 분포가 평평할 때만 ensemble lift 살아남는 환경 존재**.

**핵심 결론**: "ensemble lift 가 보장되지 않는 환경 존재" — production 에서 ensemble 추가 시 항상 weight tuning + member quality 균형 검증 필요. Score-blind aggregation (RRF) 보다 score-based learn-to-rank (LGBM) 이 robust.

---

## exp_010_lgbm_reranker — DONE 2026-05-26 (제출 #7, 0.1353)

[code](./exp_010_lgbm_reranker/) · Two-stage architecture: 5-model stage-1 candidates (top-50 union) → 5-fold LGBM rerank.

### 구조

- **Stage 1 candidate pool**: top-50 union of {tifu, bsarec, mbstr, bsarec_cl, bert4rec} = **74.5M (user, item) pairs**, 638k users
- **Features (59 cols, 51 numeric)**: 
  - A. model_ranks (5 models × {rank, norm_score, in_top10} + n_models_in_top10): 18 cols
  - B. item_popularity (view/cart/purchase count, days_first/last, unique_users, avg_price, log1p variants): 13 cols
  - C. user_activity (total/distinct items, view/cart/purchase, recency, repeat_ratio, ratios): 15 cols
  - D. user_item_history (per (u,i) total + ev_type + days_since_last): 5 cols
  - E. user_item_affinity (brand/category top1 match): 2 cols + 4 categorical (excluded from training)
- **Stage 2 LGBM**: 5-fold by user_id on 928 eval_users (binary logloss, num_boost=1000, early_stopping=50). num_leaves=63, max_depth=7, min_data=50, feature_frac=0.8, bagging_frac=0.8.
- **Inference**: 5-fold ensemble (avg predictions), top-50 per user across 638k users → output.csv

### 결과

| Metric | LGBM (exp_010) | TIFU baseline (exp_007) | Δ |
|---|---:|---:|---:|
| OOF NDCG@10 (eval 928명) | **0.3418** | 0.2922 | **+0.0496** |
| OOF recall@10 | 0.4258 | 0.3915 | +0.0343 |
| Public NDCG@10 | **0.1353** | 0.1175 | **+0.0178 (+15.2%)** |

**Calibration ratio**: 0.342 / 0.1353 = **2.527** — TIFU 2.49 / BSARec 2.53 사이. **Family-agnostic framework 추가 검증** (이제 4종 measurement: ALS 2.32 / TIFU 2.49 / BSARec 2.53 / LGBM 2.527).

### Top 20 features by gain

```
tifu_rank             9085   ← 압도적 (TIFU 가 backbone)
ui_days_since_last    2701   ← user-item history recency
tifu_norm_score       2342
item_cart             2162   ← popularity (cart events 가 purchase 보다 강한 signal)
user_repeat_ratio     1897
item_days_since_last  1798
user_days_recency     1438
user_days_since_first 1410
item_days_since_first 1386
bert4rec_rank         1250   ← LOO-overfit 모델도 LGBM 은 활용
item_total_events     1232
item_avg_price        1206
bsarec_norm_score     1173
item_unique_users      972
bsarec_cl_norm_score   959   ← BSARec+CL 도 신호 기여
bsarec_rank            864
bert4rec_norm_score    796
mbstr_norm_score       763   ← v1 (hidden=64) 도 의미 있는 기여
user_total_events      617
user_distinct_items    603
```

### 핵심 학습

1. **Two-stage architecture (candidate gen → reranker) effectiveness 직접 입증**:
   - 5 score-blind RRF / weighted ensemble 다 fail
   - LGBM (score-based learn-to-rank) 으로 pivot → **same models 로 +0.0178 public lift**
   - **"모델 추가보다 reranker 가 더 큰 lever"** — production 표준 architecture 의 mechanism 확인

2. **Reranker 가 LOO-overfit 모델 (BERT4Rec, BSARec+CL) 도 활용 가능**:
   - RRF 에서는 LOO-overfit 가 ensemble 오염 (v7 가설 H 의 결론 와 모순 아님 — RRF 와 LGBM 의 메커니즘 차이)
   - LGBM 은 feature importance 통해 자동 weight tuning → 약한 모델도 marginal signal 활용
   - **Score-based learn-to-rank vs score-blind aggregation 의 차이 사례**

3. **Feature backbone: TIFU rank dominant + popularity/recency 보조**:
   - tifu_rank gain 9085 vs 다음 ~2700 → 3.3배 dominant. TIFU 가 stage-1 backbone 임을 정량 확인
   - item_cart > item_purchase 인 게 흥미로움 (cart 가 추후 purchase 의 더 강한 leading indicator)
   - user_repeat_ratio (#5) 강함 — repeat-buyer 가 dominant population

4. **Memory engineering (OOM 4번 → chunked parquet pipeline)**:
   - container cgroup ~60GB. 초기 build_features.py 가 74M-row pandas DataFrame + intermediate merge 로 56GB anon-rss → killed
   - Fix: per-chunk parquet 으로 [D] history + [E] affinity + label 통합. features_all 이 디렉토리 (8 × ~340 MB)
   - inference.py 도 same pattern (per-chunk predict + top-50 trim)
   - **Production engineering signal**: cgroup / chunked pipeline / dtype downcast (int16, float32) / del + gc

5. **Label leakage 자체 발견**:
   - inference.py 의 self-val 0.561 출력 → 의심 → fold 모델 각각 80% eval_users 학습 → 5-fold ensemble 적용 시 4-out-of-5 가 user 본 상태 → leakage
   - **OOF NDCG 0.342 가 honest estimate** 이라고 정정
   - Calibration ratio 도 OOF 기준 계산 → 2.527 (실제 public 과 정합)
   - **ML correctness 감각** — junior 에서 쉽게 놓치는 함정

### Lift mechanism

**+15.2% lift 의 의미**:
- 동일 stage-1 모델 set 위에 **score-blind RRF/weighted ensemble 모두 fail** (v3-v7)
- 같은 모델 set + score-based learn-to-rank (LGBM) → **+0.0178 absolute lift**
- 모델 추가 없이 reranker 도입만으로 base TIFU 의 **+15.2% lift**

**Narrative**: "**Single-model base + strong reranker** — 한 모델만 운영 가능한 production 환경에서 reranker pivot 의 효과를 동일 dataset 으로 입증. 모델 추가 비용 0 으로 retrieval → ranking 2-stage architecture 의 mechanism 직접 확인."

### 한 줄 요약

> "5개 stage-1 모델 (classical TIFU + transformer 4종) 의 score-blind RRF/weighted ensemble 4가지 방식 모두 single best TIFU 못 넘었다. 진단으로 LOO-overfit 패턴 + RRF aggregation 한계 발견. **2-stage LGBM reranker 로 pivot** 해서 same models 로 public **+15.2% lift** 달성. 이 과정에서 production engineering 패턴 (cgroup-aware chunked parquet pipeline) 직접 구현하면서 OOM 4번 해결. 마지막 label leakage 함정도 self-val 0.561 출력 보고 직접 catch — OOF 0.342 가 honest estimate."

---

## exp_010b_lgbm_lambdarank — DONE 2026-05-27 (제출 #8, 0.1358)

[code](./exp_010b_lgbm_lambdarank/) · exp_010 의 LGBM objective swap. **Binary classifier → LambdaRank**. Features rebuild 불필요 (exp_010 의 features_all/ 재사용).

### 가설 + 동기

exp_010 의 `LGBMClassifier(objective="binary")` 는 NDCG@10 metric 과 objective misalign. LightGBM 의 `LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[10])` 가 직접 정합. 가설: **LambdaRank 로 swap 시 OOF NDCG +0.005-0.010 lift, public +0.003-0.007 추정**.

### 변경점

- `objective`: binary → **lambdarank**
- `metric`: binary_logloss → **ndcg, eval_at=[10]**
- `lgb.train` → `LGBMRanker.fit(group=per_user_row_count)`
- Stage 1 candidate pool, feature engineering, 5-fold split 모두 동일

### 결과

| Metric | exp_010 binary | **exp_010b LambdaRank** | Δ |
|---|---:|---:|---:|
| OOF NDCG@10 | 0.3418 | **0.3513** | **+0.0095** |
| OOF recall@10 | 0.4258 | 0.4369 | +0.0111 |
| Public NDCG@10 | 0.1353 | **0.1358** | **+0.0005** |

**Fold-wise 비교** (fold 0-3, lambdarank 가 모든 fold 에서 binary 보다 ↑):

| Fold | Binary | LambdaRank | Δ |
|---|---:|---:|---:|
| 0 | 0.3443 | 0.3489 | +0.0045 |
| 1 | 0.3037 | 0.3242 | **+0.0205** |
| 2 | 0.3449 | 0.3568 | +0.0119 |
| 3 | 0.3149 | 0.3216 | +0.0067 |

**Calibration ratio 변화**:
- exp_010 binary: 0.3418 / 0.1353 = **2.527**
- exp_010b LambdaRank: 0.3513 / 0.1358 = **2.587** (+2.4%)

### 핵심 lesson — OOF lift 의 public 전이 불충분

**예상 vs 실제**:
- 예상 public lift: +0.005 ~ +0.007 (calibration ratio 2.527 가정)
- 실제 public lift: **+0.0005** (예상의 5-10% 만 실현)

**가능한 원인**:

1. **LambdaRank 의 per-user objective 가 small eval (928) 에 over-fit**:
   - Binary: 100k rows 의 global logloss → 더 robust
   - LambdaRank: per-user group 의 ranking → small group 의 noise 학습 가능

2. **OOF measurement 자체의 inflation**:
   - LambdaRank 가 NDCG 직접 최적화하니 OOF NDCG 가 더 sensitive 하게 inflate
   - Public test (Mar 1-7) distribution 과 mismatch

3. **Validation 설계 한계**:
   - eval_users 928 + val_gt 1,443 (purchase only) sample 작음
   - 더 큰 sample (e.g., cart+purchase) 사용했다면 LambdaRank lift 더 안정적일 가능성

### 한 줄 요약

> "exp_010 binary LGBM 의 NDCG@10 misalign 진단 후 LambdaRank 로 swap. OOF +0.0095 lift 명확하게 나왔지만 **public 은 +0.0005 만 전이**. Calibration ratio 가 objective 별로 다름 (2.527 → 2.587) 을 정량 발견. Lesson: **OOF improvement 가 public improvement 보장 X**, 특히 per-user ranking objective 는 small eval sample 에 over-fit 위험. Production 의 'offline metric 좋은데 online 별로' 패턴의 mini 사례."

### 산업 적용

- **Self-val sample size 가 ranking objective 의 reliability 결정**
- Production: A/B test 가 ground truth, offline NDCG 는 proxy
- Per-user ranking 은 user 당 충분한 sample (e.g., 행동 5+ events) 있을 때 안정적

---

# 최종 pivot 결정 (2026-05-27)

**Week 1 모델링 종료. Week 2 service 정리 시작**.

## 모델링 결산

- 7개 모델 (EASE, ALS, BSARec(4종 window), BERT4Rec, BSARec+CL, TIFU, TIFU multi-behavior, MB-STR v1/v2/v2b)
- 6 종 ensemble + 1 LGBM reranker
- 진단 노트북 1개 (TIFU vs BSARec, 5 segment 분석)
- 제출 7회, 최고 public 0.1353

## Lift grinding ROI 종료 근거

| 후보 path | 시간 | 예상 lift | 확률 |
|---|---:|---:|---:|
| MB-STR v2b 완주 + features rebuild + LGBM rerun | ~7h | +0.002 public | 30% (v2 plateau pattern 우려) |
| LGBM ablation (drop user_activity 등) | 1h | +0.001-0.003 | 60% |
| TIFU multi-behavior weight 튜닝 | 2-4h | +0.005 | 40% |
| **합쳐서** | **~10h** | **+0.005-0.01** | **~50%** |

10h 투자 후 expected lift +0.005-0.01. 추가 grinding 의 marginal 효용 < Week 2 service deliverable 의 가치 → **Lift grinding 종료, service 트랙 진입**.

## Week 2 service ROI (대안)

- FastAPI 추천 API: 핵심 deliverable
- LLM 통합 (Solar Pro2 / OpenAI-compatible): "추천 + 설명 generation"
- 트레이드오프: "**모델 0.005 추가 lift vs FastAPI + LLM demo 영상**" → demo 가 ROI 훨씬 큼
- 본인 2026-05-21 pivot 결정 (memory) 와 정합

## 컨벤션 — 새 실험 추가 시

1. `experiments/exp_NNN_<name>/` 코드+config 만. README 안 만듦
2. 이 파일 leaderboard + 제출 이력 + 실험 섹션 (가설/결과/학습) 추가
3. [docs/references.md](../docs/references.md) 에 새 모델/라이브러리 row 추가
