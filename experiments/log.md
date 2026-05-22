# Experiments Log

모든 실험의 단일 lab notebook. 각 실험 폴더에는 코드+config만 두고, **가설/결과/학습은 이 파일에 작성**.

**대회**: Commerce Behavior Purchase Prediction · **Train**: 2019-11-01 ~ 2020-02-29 (8.35M events, 638,257 users × 29,502 items) · **평가**: 2020-03-01 ~ 2020-03-07, NDCG@10 binary, public/private 50:50
**제출 규정**: 동점 시 제출 횟수 적은 쪽 우위 → **무의미한 제출 회피**

**자체 val 규약**: train 마지막 7일 (Feb 23-29) hold-out + `restrict_to_train=True` + `gt_event_types=['purchase']` + `eval_users` 928명. exp_001 의 `val_gt.parquet` + `eval_users.json` 을 후속 실험 모두 재활용.

**Calibration**: self-val NDCG@10 ÷ 2.53 ≈ public (sequential 기준, exp_002/002b/002e 3회 검증, 오차 ±0.0005). ALS 만 2.32. val 의 99.7% 가 Feb 27-29 spike → self-val 은 사실상 spike 예측 skill만 측정.

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
| **exp_002e** | **BSARec (4w_full, spike+)** | **sequential** | **0.2470\*** | **0.3274\*** | **0.0975** | **DONE — top** |
| exp_002g | BSARec (2w_full, spike+) | sequential | 0.2479\* | 0.3304\* | 0.0975 | DONE (tie with 002e) |
| exp_002f | BSARec (1w_full, spike만) | sequential | 0.2408\* | 0.3190\* | — | DONE (less data hurts) |
| exp_003 | DiffRec | diffusion | 0.1543 | 0.2257 | — | DONE (< ALS, paradigm coverage 만) |
| exp_004 | FEARec | sequential FFT+autocorr | — | — | — | FAILED (RecBole 1.2 hang) |
| exp_005 | BERT4Rec | bidirectional MLM | TBD | TBD | — | RUNNING |
| exp_006 | BSARec+CL4SRec hybrid | sequential + contrastive | — | — | — | QUEUED (novel) |
| exp_007 | TIFU-KNN | classical KNN + temporal freq | — | — | — | QUEUED (paradigm coverage, classical vs deep 진단) |
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
| 4 | 2026-05-22 | exp_002e BSARec 4w_full | **0.0975** | spike 포함 retrain 효과. 예측 0.0977 vs 실제 0.0975. 새 leaderboard 1위 |
| 5 | 2026-05-22 | exp_002g BSARec 2w_full | 0.0975 | 002e 와 tie. tie-break 상 002e 가 002g 보다 상위 (제출 횟수) |

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

## exp_005_bert4rec — RUNNING (Day 1B)

[code](./exp_005_bert4rec/) · **CIKM 2019** bidirectional MLM Cloze.

**가설**: BSARec/SASRec 의 causal next-item 과 다른 학습 신호 (bidirectional context). Sequential family 내 second signal.

**하이퍼**: n_layers=2, hidden=64, n_heads=2, mask_ratio=0.2, max_seq=50, batch=2048, lr=0.001. eval_step=2.

**진행 (2026-05-22 05:30)**: epoch 0 484s (8 min), epoch 1 484s. **epoch 1 RecBole LOO NDCG 0.1762**, recall 0.3087. epoch 2 진행 중.

**결정점**: epoch 3 eval (~12분 후) 보고 — 0.18+ 면 계속, 0.17 정체 면 kill 후 exp_006 전환.

---

## exp_007_tifu_knn — QUEUED (Day 1B+, classical baseline)

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

**제출 전략**:
- 단독 self-val 결과 진단 → 0.10+ 면 paradigm coverage 성공, 0.05+ 면 ensemble 후보로만
- 점수 chasing 아님 — "classical vs deep" 진단이 portfolio value 핵심

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

| Day | Exp | Window | 상태 | 비고 |
|---|---|---|---|---|
| 1A | exp_004 FEARec | — | FAILED | RecBole 1.2 hang |
| 1B | exp_005 BERT4Rec | **4m** | RUNNING | self-supervised MLM → sample 다양성 우위 |
| 1B+ | exp_007 TIFU-KNN | **4m** | QUEUED | classical paradigm coverage. CPU 작업 (BERT4Rec GPU 안 충돌) |
| 1C | exp_006 BSARec+CL hybrid | **4w** | QUEUED | BSARec recency 검증됨 + CL aug 은 data 양 무관 |
| 2-3 | MB-STR | **4m** | 예정 | multi-behavior: rare purchase 빈도 보존 필수 |
| 4 | LLM-as-Reranker | base 따름 | 예정 | Anthropic API, top-50 contextual rerank |
| 5 | ensemble v3 + LGBM reranker | 4m+4w 후보 혼합 | 예정 | weighted RRF, 6+ models, writeup |
| 5 final | Winner retrain (val 포함) | model-specific | 예정 | 4w-winner → 4w_full / 4m-winner → 4m_full |
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
