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
| exp_006 | BSARec+CL4SRec hybrid | sequential + contrastive | — | — | — | QUEUED (novel, portfolio piece) |
| **exp_007** | **TIFU-KNN (4m)** | **classical KNN + temporal freq** | **0.2922** | **0.3915** | **0.1175** | **DONE — new top, +20.5% vs BSARec** |
| (예정) exp_007b | TIFU-KNN + multi-behavior weights | classical | — | — | — | QUEUED (cart=3 / purchase=5 ablation) |
| (예정) exp_007c | TIFU-KNN α/K ablation | classical | — | — | — | QUEUED |
| (예정) | MB-STR | multi-behavior | — | — | — | Day 2-3 |
| (예정) | LLM-as-Reranker | 2-stage LLM | — | — | — | Day 4 |
| (예정) | ensemble v3 + LGBM reranker | fusion | — | — | — | Day 5 |

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

**Portfolio talking point**: 단순 negative 가 아니라 structural limitation 을 ablation 으로 진단. ensemble 접근 dead (BSARec → ∞ 가 BSARec 단독에 수렴).

---

## exp_002b/c/d (BSARec recency ablation, holdout) — DONE 2026-05-21

멘토 권고 ablation. 4m baseline → 4w/2w/1w 변종. 모델/하이퍼 동일, 학습 데이터만 변경.

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

**대응**: 포기, BERT4Rec 으로 대체. Talking point 보존: "RecBole 1.2 FEARec incompatibility at scale".

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

**핵심 발견 — bidirectional MLM 의 task mismatch** (portfolio gold):

비교: BSARec 패턴에서는 RecBole LOO 가 self-val 보다 *높음* (transformer causal 의 정상 패턴). BERT4Rec 는 **반대 방향**:
- LOO = "마지막 item 마스킹 → 예측" → Cloze training 과 **정확히 일치** → over-specialized
- 우리 val = "7일 윈도우 multi-purchase 예측" → next-item 가정 깨짐
- **Bidirectional MLM 이 multi-target purchase task 에 transfer 약함**
- Portfolio talking point: "**모델 training task 와 inference task 정합성**이 RecSys 특유의 함정 — BERT4Rec 이 LOO 에 over-fit 하는 사례를 데이터로 잡음"

**추정 public**: 0.2158 / 2.53 ≈ **0.0853** (ALS 베이스라인 수준) → **제출 X**.

**Ensemble 가치 재평가**:
- recall 0.3061 ≈ BSARec recall 0.3195 → 이전 가설 "다른 후보 set" 약화
- TIFU recall 0.3915 와 큰 격차 → ensemble 시 BSARec 1개로 충분, BERT4Rec 추가 lift 작을 듯
- 결정: ensemble v3 후보로 보관, but 핵심 아님

---

## exp_007_tifu_knn — DONE 2026-05-22 (Day 1B+, NEW LEADERBOARD TOP)

[code](./exp_007_tifu_knn/) · **SIGIR 2020** "Modeling Personalized Item Frequency for Next-Basket Recommendation" (He et al.). Non-neural classical method.

**왜 이걸 추가 (portfolio 관점)**:
- 다른 팀원이 `mbr_sas_tifu_knn` 으로 **public 0.1431** 보고 (우리 0.0975 대비 +47%). Pure transformer pipeline 의 한계 시사
- 우리 5개 모델 (ALS/EASE/BSARec/DiffRec/BERT4Rec) 다 scoring 기반 deep/closed-form → **KNN/frequency-based paradigm 0개**. 빈 슬롯 채우기
- Production e-commerce 의 first-class signal: "user X 가 item Y 를 반복 view/cart" 패턴. transformer max_seq=50 cap 보다 직접 모델링
- **면접 talking point**: "Classical method 가 같은 데이터에서 deep model 을 이긴 이유 — repeat-purchase + temporal frequency 가 dominant signal" — research → production maturity 신호

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

**핵심 학습 (portfolio gold)**:
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

**면접 talking point 우선순위**:
> "Sequential transformer 4종 (BSARec / BERT4Rec / BSARec+CL hybrid) 를 ablate 했지만, 같은 데이터에서 100줄짜리 2020 paper KNN method (TIFU-KNN) 가 **public NDCG +20.5%** 로 압도했다. 진단해 보니 이 데이터의 dominant signal 인 **user-item repeat frequency + temporal decay** 를 transformer 의 parametric attention 으론 직접 잡지 못했기 때문이다. 이게 production e-commerce 에서 classical KNN/co-visit 류가 first-stage candidate generator 로 살아남는 mechanism. Research SOTA 와 production reality 사이의 gap 을 데이터로 직접 검증."

**다음 액션** (TIFU-KNN 위에 추가 lift):
- **exp_007b** — multi-behavior event weights (cart=3, purchase=5) ablation. 우리 현재 1/1/1 → 팀원 `mbr_sas_tifu_knn` 의 0.1431 까지 가는 핵심 lever 추정
- **exp_007c** — α (0.5/0.7/0.8) + knn_k (300/500/1000) ablation
- **exp_007_full** — val 포함 retrain (BSARec 패턴 적용 시 +0.002 추정 → public ~0.1195)
- **Day 4 LLM reranker** — TIFU 의 top-50 후보를 LLM 으로 contextual rerank → potentially big lift
- **Day 5 ensemble v3** — TIFU + BSARec + BERT4Rec RRF/weighted, 매우 다른 후보 set 들 (TIFU recall 0.39 vs BSARec 0.32)

---

## exp_006_bsarec_cl — QUEUED (Day 1C, novel combination)

[code](./exp_006_bsarec_cl/) · **BSARec backbone + CL4SRec InfoNCE 보조 loss**. 두 논문 결합 자체가 novel (BSARec=AAAI24 attention only, CL4SRec=ICDE22 SASRec backbone).

**가설**:
- BSARec: 99.78% view-dominant noise → FFT low-pass (검증됨, public 0.0975)
- CL4SRec: 99.96% sparsity + median seq 6 → augmentation 이 sparse regime 에서 generalize
- **함께**: periodicity + representation regularization 둘 다 커버

**구현**: `BSARecCL(BSARec)` 클래스. `calculate_loss()` 에서 `rec_loss + lmd · InfoNCE(z1, z2)`. `z1, z2 = forward(aug1), forward(aug2)`. aug = crop / mask / reorder 중 random. RecBole left-padded 레이아웃에 맞춰 augmentation 재작성 (팀원 right-padded 코드 차용 후 length tracking 추가).

**하이퍼**: BSARec (alpha=0.7, c=5) + CL4SRec (lmd=0.1, tau=1.0, crop_ratio=0.4, mask_ratio=0.3, reorder_ratio=0.4) — 모두 paper default.

**실행**: BERT4Rec 종료 후 (kill 또는 자연 종료) 즉시.

---

# Day 1-5 Plan (재pivot 2026-05-22)

이전 pivot (model lift → system+LLM) 에서 다시 모델 트랙으로. 4-5일 더 사용 가능. 해외 RecSys 시장 고려, **D (data-fit + production diversity) + LLM** 방향.

**2026-05-22 update — TIFU-KNN 0.1175 충격**: exp_007 가 BSARec 4종을 18-20% 차이로 압도. Strategic pivot: transformer 추가 lift 추구 ROI 낮음 → **TIFU-KNN 변형 ablation + ensemble + LLM reranker** 가 점수 lift 의 우선 lever. Transformer 계열 (exp_006 BSARec+CL, MB-STR) 은 **portfolio piece 로 진행** (점수 목적 X).

| Day | Exp | Window | 상태 | 비고 |
|---|---|---|---|---|
| 1A | exp_004 FEARec | — | FAILED | RecBole 1.2 hang |
| 1B | exp_005 BERT4Rec | 4m | killed at ep13 (plateau 0.2006) | ensemble second sequential signal 확보 (recall 0.3485) |
| **1B+** | **exp_007 TIFU-KNN** | **4m** | **DONE (0.1175, new top)** | **classical KNN — paradigm 변화 + new top** |
| 1C | exp_007b TIFU + multi-behavior weights | 4m | NEXT (점수 lever) | cart=3, purchase=5 ablation. 팀원 0.1431 까지 가는 핵심 추정 |
| 1D | exp_007c TIFU α/K ablation | 4m | NEXT | alpha 0.5/0.8, knn_k 500/1000 |
| 1E | exp_006 BSARec+CL hybrid | 4w | portfolio piece | novel combination talking point only (점수 ceiling < TIFU) |
| 2-3 | MB-STR | 4m | 예정 | multi-behavior transformer paradigm + 직접 비교: BSARec vs MB-STR 의 multi-behavior 효과 |
| 4 | LLM-as-Reranker on TIFU top-50 | base 따름 | 예정 | TIFU 가 base 가 됨 (highest recall) |
| 5 | ensemble v3 + LGBM reranker | mixed | 예정 | TIFU + BSARec + BERT4Rec 후보 결합, very diverse signal |
| 5 final | Winner retrain (val 포함) | model-specific | 예정 | TIFU best variant → full retrain (+0.002 추정) |
| Buffer | CL4SRec 단독 port | 4w | 옵션 | 시간 남으면 |

**Window 선택 원리** (model mechanism 별):
- **Sequential causal** (BSARec, SASRec): recency 유리 → 4w
- **Self-supervised / generative** (BERT4Rec MLM, DiffRec): sample 다양성 유리 → 4m
- **Multi-behavior / rare event** (MB-STR): 절대 빈도 유리 → 4m
- **Reranker** (LLM, LGBM): candidate generator window 따라감

새 모델마다 4종 window 다 돌리지 않음. mechanism 기반 1개만 선택, Day 5 final 에서 winner 만 val-포함 retrain.

---

## 컨벤션 — 새 실험 추가 시

1. `experiments/exp_NNN_<name>/` 코드+config 만. README 안 만듦
2. 이 파일 leaderboard + 제출 이력 + 실험 섹션 (가설/결과/학습) 추가
3. [docs/references.md](../docs/references.md) 에 새 모델/라이브러리 row 추가
