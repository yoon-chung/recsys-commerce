# Experiments Log

본 프로젝트의 모든 실험을 시간순으로 기록하는 단일 lab notebook. 각 실험 폴더 (e.g. `exp_001_ease/`) 에는 코드 + config만 있고, **가설 / 하이퍼 / 결과 / 결론은 모두 이 파일에 작성**.

**대회**: Commerce Behavior Purchase Prediction
**Train**: 2019-11-01 ~ 2020-02-29 (4개월, 8.35M events, 638,257 users × 29,502 items)
**평가**: 2020-03-01 ~ 2020-03-07 (1주), NDCG@10 binary, public/private 50:50
**제출 규정**: 동점 시 제출 횟수 적은 쪽 우위 → **무의미한 제출 회피**

**자체 val 규약**: train 의 마지막 7일 (Feb 23-29) hold-out + `restrict_to_train=True` + `gt_event_types=['purchase']` + `eval_users` 928명. exp_001 의 `saved/val_gt.parquet` + `eval_users.json` 을 후속 실험 모두 재활용 (fair comparison).

**Calibration**: self-val NDCG@10 ÷ 2.32 ≈ public NDCG@10 (ALS + 113일 학습 기준 측정값. 다른 모델 / split 에서는 비율 달라질 수 있음).

---

## 베이스라인 참조

| 모델 | Public NDCG@10 | 비고 |
|---|---:|---|
| ALS (주최사) | 0.0847 | `/root/code/train_als.py` |
| SASRec (주최사) | 0.0842 | `/root/code/train_sasrec.py` |

---

## 결과 leaderboard (live)

| Exp | Model | Family | Val NDCG@10 | Val recall@10 | Public | Status |
|---|---|---|---:|---:|---:|---|
| exp_000 | ALS | MF | 0.1838 | 0.2558 | **0.0791** | DONE |
| exp_001 | EASE | item-item | 0.1848 | 0.2909 | (제출 X) | DONE |
| ensemble_v1 | ALS+EASE RRF | fusion | 0.1725 | 0.2624 | (제출 X) | DONE (negative) |
| **exp_002** | **BSARec (4m)** | **sequential** | **0.2391** | **0.3195** | **0.0943** | **DONE** |
| ensemble_v2 (equal) | ALS+EASE+BSARec RRF | fusion | 0.2067 | 0.3207 | (제출 X) | DONE (negative −0.032 vs BSARec) |
| ensemble_v2 (w 1:1:2) | weighted RRF (BSARec 2x) | fusion | 0.2031 | 0.3068 | (제출 X) | DONE (negative −0.036, **worse than equal**) |
| exp_002c | BSARec (2w) | sequential | TBD | TBD | — | QUEUED |
| exp_002d | BSARec (1w) | sequential | TBD | TBD | — | QUEUED |
| **exp_002b** | **BSARec (4w)** | **sequential** | **0.2414** | **0.3242** | **0.0955** | **DONE — +0.0012 public vs 4m (recency 효과 실증)** |
| exp_003 | DiffRec | diffusion | TBD | TBD | — | QUEUED |
| (예정) | LightGCN | graph | — | — | — | Week 1 Day 6 |

---

## 제출 이력

| # | Date | Exp | Model | Public NDCG@10 | 비고 |
|---:|---|---|---|---:|---|
| 1 | 2026-05-20 | exp_000 | ALS | **0.0791** | calibration. 베이스라인 공시 0.0847 대비 6.6% gap ≈ 7/120일 holdout 비율. 파이프라인 정상 확인 |
| 2 | 2026-05-21 | exp_002 | BSARec | **0.0943** | **+11.3% vs ALS 베이스라인, +12.0% vs SASRec 베이스라인**. self-val 0.2391 / public 0.0943 = ratio **2.535** (ALS 2.32 보다 ~9% 큼 — sequential 이 self-val 더 inflate). Family-diverse ensemble v2 의 기반 |
| 3 | 2026-05-21 | exp_002b | BSARec (4w) | **0.0955** | **+12.8% vs ALS / +13.4% vs SASRec 베이스라인**. self-val 0.2414 / public 0.0955 = ratio **2.528** (4m 2.535 와 사실상 동일). Δ vs 4m: self-val +0.0023 / public +0.0012. **calibration 모델 예측 정확도 검증**: 예측 0.0953 vs 실제 0.0955 (오차 0.0002) |

---

## 제출 전 체크리스트

`core.submission.validate_submission(csv_path)` 가 자동 검증해주는 것 외에 사람이 한 번 더 확인:

- [ ] `output.csv` shape == (6,382,570, 2), 헤더 `user_id,item_id`
- [ ] user 638,257 전원 + user당 10개 (validate 통과)
- [ ] user당 item 중복 0건 (validate 통과)
- [ ] **자체 val NDCG@10 이 베이스라인 (÷2.32 → 0.0847 라인) 대비 명확히 높음** 또는 ensemble 다양성 목적 명확
- [ ] `predictions.parquet` (top-50 + score) 도 함께 저장 → ensemble 입력 가능
- [ ] wandb artifact 업로드 (모델 + predictions)
- [ ] 이 파일 leaderboard + 제출 이력 + 해당 실험 섹션 업데이트
- [ ] `git status` — 데이터/베이스라인/raw ID 파일이 staging 에 없는지

---

# 실험 기록

가장 오래된 것부터 시간순 stacking. 각 실험은 (가설 → 하이퍼 → 실행 → 결과 → 결론 → 다음액션) 구조.

---

## exp_000_als_baseline — DONE (2026-05-20)

[code: ./exp_000_als_baseline/](./exp_000_als_baseline/) · 출처: [references §1 ALS](../docs/references.md#1-추천-모델--논문--공식-코드)

### 가설 (목적)

점수 향상이 아니라 (1) reference 점수 확보 (2) `core/` 인프라 검증 (3) self-val ↔ public 환산 비율 측정.

### 접근

- `implicit.als.AlternatingLeastSquares` (GPU)
- Train: train_df (event_time < cutoff=last 7d) view/cart/purchase. (user,item) 합산 confidence
- Universe: train 등장 user/item 만 (623,866 / 29,413). cold-start ~14k 는 submission 시점 popularity_fallback
- Eval: 마지막 7일 purchase = ground truth (1,443 rows / 1,105 users / 928 eval after restrict)
- 베이스라인 코드 참조 금지 — implicit 컨벤션 따라 직접 구현

### 하이퍼파라미터 (베이스라인 매칭)

| 항목 | 값 | 베이스라인 |
|---|---|---|
| factors / regularization / alpha | 32 / 0.001 / 10 | 동일 |
| iterations / seed | 15 / 42 | 동일 |
| event_weights (view/cart/purchase) | 1 / 1 / 1 | 동일 |
| filter_already_liked | **false** | 동일 (핵심 flag) |
| use_gpu | true | 베이스라인 False (GPU 만 다름) |

### 결과

| 메트릭 | 값 | 비교 |
|---|---:|---|
| 자체 val NDCG@10 | **0.18376** | 1차 mismatched 0.0288 대비 6.4× |
| 자체 val recall@10 | **0.25578** | 1차 0.0566 대비 4.5× |
| Public NDCG@10 | **0.0791** | self-val 의 43%. 베이스라인 공시 0.0847 대비 6.6% 낮음 |

> **1차 시도** (mismatched config): factors=128 / reg=0.01 / alpha=40 / weights 1·3·5 / `filter_already_liked=true`. self-val 0.0288 (1/6). 학습 차이 아닌 **config 차이가 원인**.

### 결론

1. **파이프라인 정상** — 공시 갭 6.6% ≈ 학습 데이터 손실 5.8% (7/120일). 알고리즘/구현/core 모두 등가
2. **calibration 2.32 확정** — self-val ÷ 2.32 = public 기대치. (다른 모델/split 에서 달라질 수 있음, sequential 후 한 번 더 calibration 권장)
3. **self-val 절대값은 Feb 27-29 spike 로 inflate** ([eda §5](../docs/eda_findings.md)). 같은 split 내 상대 비교만 신뢰
4. **`filter_already_liked=false` 가 6.4× 점프의 주요 lever 의심** (ablation 안 함). Feb 27-29 spike 의 "직전 view → 구매" 패턴 차단 안 한 게 핵심

### 다음 액션

- ✅ calibration 제출 완료
- 다음: sequential 모델 ([exp_002 BSARec](#exp_002_bsarec--running-2026-05-21))

---

## exp_001_ease — DONE (2026-05-21)

[code: ./exp_001_ease/](./exp_001_ease/) · 출처: [references §1 EASE](../docs/references.md#1-추천-모델--논문--공식-코드)

### 가설

- **EDA 정합**: item 반복도 14.6% + 99.78% view-dominated → item-item similarity 시그널 강력
- **Ensemble 다양성**: ALS (MF factor) vs EASE (item-item) → RRF lift 기대
- **서비스 친화적**: inference = sparse × dense 1회 (1 user < 1ms). Week 2 service 데모 적합

### 알고리즘 (30줄 closed-form)

```
G = X^T X                     # item × item Gram
P = inv(G + λI)               # Cholesky factor + cho_solve (PD 행렬)
B = -P / diag(P)[None, :]     # 행별 정규화
fill_diagonal(B, 0)           # self-loop 제거
score = X @ B
```
복잡도: O(n_items³) inversion. peak 메모리 ~14GB (서버 251GB OK, 로컬 16GB 빠듯).

### 하이퍼파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| `reg_lambda` | 200 | paper MovieLens 100-1000. B stats 양호 (max=2.07, p99=4.9e-3) → ablation 우선순위 낮음 |
| `event_weights` | 1/1/1 | exp_000 동일 (fair) |
| `filter_already_liked` | false | exp_000 lesson |
| `dtype_compute / storage` | float64 / float32 | inversion 안정성 + 저장/추론 |

### 결과 (2026-05-21)

| 메트릭 | EASE | exp_000 ALS | 차이 |
|---|---:|---:|---:|
| 자체 val NDCG@10 | **0.1848** | 0.1838 | +0.5% |
| 자체 val recall@10 | **0.2909** | 0.2558 | **+13.7%** |
| Public NDCG@10 (추정) | ~0.0797 (÷2.32) | 0.0791 (실측) | ~동일 |

학습 ~5분 (Cholesky 38s + cho_solve 162s), 추론 ~9분.

### 결론

NDCG 비등 + recall 큰 차이 → EASE 는 정답을 top-10 에 **더 자주** 잡지만 순위는 ALS 와 비슷. item-item 시그널이 강한 데이터. 제출은 안 함 (calibration ratio 적용 시 ALS public 과 동일 예상).

### 다음 액션

- → [ensemble_v1](#ensemble_v1_als_ease--done-2026-05-21-negative) 으로 ALS+EASE RRF 검증

---

## ensemble_v1_als_ease — DONE (2026-05-21, negative)

[code: ./ensemble_v1_als_ease/](./ensemble_v1_als_ease/) · 출처: [references §1 RRF](../docs/references.md#1-추천-모델--논문--공식-코드)

### 가설 (실제로는 negative 로 판명)

- RRF 가 두 모델 모두 대비 NDCG lift (recall 차이가 정답 set 다양성 시사)
- 다른 family (MF factor vs item-item) → rank 차이 → fusion 효과

### 알고리즘

```
RRF(u, i) = Σ_m  1 / (k_const + rank_m(u, i))    # k_const=60 default
top-N by RRF score
```
rank 기반 → 모델 score scale calibration 불필요. heterogeneous 모델에 안전.

### 결과

| 모델 | NDCG@10 | recall@10 |
|---|---:|---:|
| exp_000 ALS | 0.18376 | 0.25578 |
| exp_001 EASE | 0.18476 | 0.29089 |
| **RRF (k=60)** | **0.17253** | **0.26243** |

Lift: NDCG vs ALS **−0.011**, vs EASE **−0.012**. recall vs EASE **−0.028** (ALS recall 만 +0.007 보충).

### 결론 — 왜 떨어졌나

1. **EASE 가 ALS dominate** — recall +13.7% (정답 set 차이). 약한 ALS 시그널 동등 가중 → noise 추가
2. **같은 family** — 둘 다 implicit-feedback CF. "다른 알고리즘" ≠ "다른 시그널"
3. **k=60 평탄** — top-50 후보에서 rank 1 vs 50 의 RRF 가중치 차이 1.8 배. individual top-rank 정답이 평균화로 밀려남
4. **recall ↑ + NDCG ↓** — fusion 으로 후보 set 다양화는 약하게 됐지만 top 으로 못 끌어올림

### 학습 가치

- **family-diverse 가 핵심**: BSARec (sequential), DiffRec (generative), LightGCN (graph) 가 단순 더 많은 CF 보다 ensemble lift 가능성 큼
- 5-모델 ensemble 에서 **per-model weight + k tuning** 필요할 수도

### 다음 액션

- ❌ submission 안 함
- → [exp_002 BSARec](#exp_002_bsarec--done-2026-05-21) 후 [ensemble_v2_als_ease_bsarec](#ensemble_v2_als_ease_bsarec--queued) 로 확장

---

## exp_002_bsarec — DONE (2026-05-21)

[code: ./exp_002_bsarec/](./exp_002_bsarec/) · 출처: [references §1 BSARec + §2 RecBole](../docs/references.md) · **AAAI 2024**

### 가설

- **다른 family 시그널**: user history 시간 순서 → ALS/EASE 가 못 잡는 영역 보완
- **FRA 효과**: 99.78% view-dominated noise → FFT low-pass smoothing 이 노이즈 완화에 유리
- **Ensemble lift 재시도**: CF + Sequential 로 family-diverse RRF 시도 가치

### 구현 — Hybrid

RecBole `SequentialRecommender` backbone 의 `TransformerEncoder` 자리에 `BSARecEncoder` 삽입. 각 layer:
```
hidden = α * FrequencyLayer(x) + (1-α) * MultiHeadAttention(x)
hidden = FeedForward(hidden)
```
- **FrequencyLayer** = `low_pass + sqrt_beta² * high_pass` after rFFT/irFFT — [fra.py](./exp_002_bsarec/fra.py) 에 직접 port (저자 코드 Apache-2.0, LayerNorm 표준화)
- **MultiHeadAttention / FeedForward**: RecBole 재사용
- **데이터/Trainer/체크포인트**: RecBole 그대로

### 하이퍼파라미터 ([config.yaml](./exp_002_bsarec/config.yaml))

| 항목 | 값 | 근거 |
|---|---|---|
| **alpha** | 0.7 | freq 비중. 논문 Beauty default |
| **c** | 5 | low-pass cutoff. 논문 sweep 1-9 |
| max_seq_length | 50 | EDA p90=29, SASRec 표준 |
| n_layers / hidden / n_heads | 2 / 64 / 2 | SASRec 표준 |
| batch / lr | 2048 / 0.002 | post-bump (initial 256/0.001 → GPU 2% 사용 → √8 스케일) |
| epochs / early-stop | 200 / step=10 | — |

### 실행

```bash
cd experiments/exp_002_bsarec
python data_prep.py                                  # 1회: ./data/cy_commerce/*.inter
nohup python train.py > train.log 2>&1 & disown      # ~1.5-2h, batch=2048
nohup python inference.py > inference.log 2>&1 & disown   # ~5min
```

### 학습 진행

| Epoch | RecBole leave-one-out NDCG@10 |
|---:|---:|
| 0 | 0.2204 |
| 8 | 0.2365 |
| 10 | 0.2381 |
| 27 | 0.2430 (first plateau) |
| 45 | 0.2433 |
| **52** | **0.2438** (best, saved) |
| 56+ | noise band (~0.243 ± 0.001), 수동 종료 |

학습 ~3시간 (batch=2048, lr=0.002). epoch 52 부터는 노이즈 lottery 단계 → 수동 kill 후 inference.

### 결과 ✅

| 메트릭 | BSARec | exp_000 ALS | exp_001 EASE | 차이 (vs EASE) |
|---|---:|---:|---:|---:|
| **자체 val NDCG@10** | **0.23910** | 0.18376 | 0.18476 | **+29.4%** |
| **자체 val recall@10** | **0.31945** | 0.25578 | 0.29089 | +9.8% |
| **Public NDCG@10** | **0.0943** | 0.0791 | (제출 X) | **+19.2% vs ALS public** |
| 학습 시간 | ~3h (batch=2048) | <1min | ~5min | — |
| n_known_users | 623,866 | 동일 | 동일 | |
| n_cold_start | 14,391 | 동일 | 동일 | |

**RecBole leave-one-out (0.2438) ↔ 우리 self-val (0.2391)** 거의 일치 — self-val 메트릭 신뢰성 부수 확인.

**Public submission 결과** (2026-05-21):
- Public NDCG@10 = **0.0943**, vs 베이스라인 ALS 공시 (0.0847) **+11.3%**, vs SASRec 공시 (0.0842) **+12.0%**
- self-val 0.2391 / public 0.0943 = **calibration ratio 2.535** (ALS 의 2.32 보다 ~9% 큼)
- 의미: sequential 모델의 self-val 이 ALS 보다 약간 더 inflate 됨. 향후 DiffRec / LightGCN 등 sequential/recency 민감 모델 추정 시 **÷ 2.5 가 더 정확**

### 결론

- ✅ **Sequential family 는 우리 데이터에서 진짜로 다른 시그널을 잡음** — [ensemble_v1 negative result](#ensemble_v1_als_ease--done-2026-05-21-negative) 의 "family diversity 가 진짜 변수" 가설을 데이터로 검증
- ✅ self-val 0.2391 은 ALS/EASE (0.185 라인) 대비 **+30%, 노이즈 범위 훨씬 초과**. 진짜 lift
- ⚠️ Public 점수가 self-val 환산값 (≈ 0.103) 에 근접할지는 sequential 모델 calibration 새로 측정 필요 — submission 필요

### 다음 액션

- **제출 강력 후보** — 베이스라인 (ALS 공시 0.0847) 명백히 초과 가능. 동시에 BSARec calibration ratio 재측정 효과
- → `ensemble_v2_als_ease_bsarec/` 생성 (3-family RRF lift 측정) — 이제 family diversity 가 진짜라서 v2 가 v1 같은 negative 결과 나올 가능성 낮음
- α/c ablation 우선순위 낮음 (이미 큰 lift 확보, marginal sweep 으로 갈 가치 작음)
- 병행: [exp_002b](#exp_002b_bsarec_4w--queued) recency window ablation (이제 더 의미 있음 — 4w 가 sequential 에 어떻게 영향?)

---

## ensemble_v2_als_ease_bsarec — DONE 2026-05-21 (negative, equal + weighted 둘 다)

[code: ./ensemble_v2_als_ease_bsarec/](./ensemble_v2_als_ease_bsarec/) · 출처: [references §1 RRF](../docs/references.md#1-추천-모델--논문--공식-코드)

### 가설 (이번엔 진짜 family-diverse)

- ensemble_v1 negative 의 근본 원인은 ALS+EASE 가 같은 CF family + EASE 가 ALS dominate. BSARec 추가로 **진짜 다른 family (sequential)** 결합
- BSARec 단독 self-val 0.2391 > ALS/EASE 0.184 라인 — 후보 set 자체가 다름
- 3-model RRF 가 BSARec 단독 (best single) 대비 **+** 면 family diversity 가설 검증

### 결과 — equal-weight RRF (k=60)

| 모델 | NDCG@10 | recall@10 |
|---|---:|---:|
| exp_000 ALS | 0.18376 | 0.25578 |
| exp_001 EASE | 0.18476 | 0.29089 |
| exp_002 BSARec | 0.23910 | 0.31945 |
| **RRF v2 equal (k=60)** | **0.20673** | **0.32071** |
| Δ vs BSARec | **−0.03237** | +0.00126 |

→ **negative NDCG (−13.5%), recall 거의 동률**. ensemble_v1 과 다른 양상이지만 결국 negative.

### 진단

- BSARec 가 NDCG 에서 압도적 (+30% vs ALS/EASE). equal-weight RRF 는 ALS/EASE 가 top rank 희석.
- recall +0.001 → fusion 이 후보 다양성은 살짝 늘림. 하지만 ordering 망가뜨림.
- ensemble_v1 (CF-only) negative = family 동질성. v2 (3-family) negative = **strength 불균형**. RRF 동등 가중 가정 위반.

### 결과 — weighted RRF (BSARec 2x, k=60)

| 시도 | NDCG@10 | recall@10 | Δ vs BSARec NDCG | Δ vs BSARec recall |
|---|---:|---:|---:|---:|
| equal 1:1:1 | 0.20673 | 0.32071 | −0.03237 | +0.00126 |
| weighted 1:1:**2** | 0.20314 | 0.30679 | **−0.03596** | **−0.01266** |

→ **weighted 가 equal 보다 더 나쁨**. recall 까지 떨어짐.

### 진단 (반직관적 결과 분석)

BSARec weight 를 올리면 dominance 가 강화되어야 하는데 왜 떨어졌나?

- RRF `1/(k+rank)` 곡선에서 BSARec rank-1 이 이미 충분히 강함. weight 2x 가 BSARec **맞은 item** 도 ↑ 시키지만 BSARec **틀린 item** 도 ↑ 시킴.
- equal 에서는 ALS/EASE 가 BSARec false positive 를 살짝 견제. 2x 가 이 견제를 약화 → recall 하락이 그 증거 (ALS/EASE 가 잡던 다른 후보가 사라짐).
- 즉 **ALS/EASE 는 보완 signal 을 가지긴 하지만, BSARec dominance 를 뒤집을 수준 X**. 어느 방향 weight 도 BSARec 단독 0.2391 을 넘기 어려운 구조.

### 최종 결론: ensemble 접근 dead

- v1 (CF-only) negative = family 동질성
- v2 equal negative = family 다양화 했지만 strength 불균형
- v2 weighted negative = dominance 살려도 BSARec false positive 도 같이 amplify

세 번 다른 가설, 모두 negative. 추가 weight sweep ROI 거의 0 (BSARec → ∞ 면 BSARec 단독에 수렴). [[project-pivot-to-system-llm]] 정신 그대로 ensemble 종료.

**Portfolio 관점**: 단순 negative 가 아니라 **structural limitation 을 ablation 으로 진단** 한 결과 — 면접 talking point 로 v1 단독보다 훨씬 강함.

### 다음 액션

ensemble 접고:
- exp_003 DiffRec (paradigm coverage, 결과 기록만), 또는
- exp_002b BSARec 4w ablation (멘토 권고), 또는
- Week 2 service 진입 ([[project-pivot-to-system-llm]] 가장 충실)

---

## exp_002b_bsarec_4w — DONE 2026-05-21

[code: ./exp_002b_bsarec_4w/](./exp_002b_bsarec_4w/) · 멘토 권고 ablation (참고: [references §4](../docs/references.md#2026-05-21-멘토링--핵심-결정))

### 가설

exp_002 와 모델/하이퍼 **완전 동일**, **학습 데이터만 마지막 4주 (Feb 1-29)**.

- **Primary**: 활성 user (4w window 안에 sequence 있는 user) 부분집합에서 4m baseline 대비 NDCG@10 lift (recency 효과)
- **Secondary**: 전체 638k 기준 NDCG 는 변동 작거나 살짝 down — 50-70% user 가 cold-start 로 빠지며 popularity fallback 비중 ↑

**근거**: BSARec sequence 는 이미 `max_seq_length=50` cap. 하지만 **item embedding 은 모든 interaction 으로 학습** → 4w 데이터 → embedding 이 Mar 1-7 분포에 더 align. + EDA Feb 27-29 spike 포함.

### 데이터 비교

| 항목 | exp_002 (4m) | exp_002b (4w) |
|---|---:|---:|
| Train rows | 8M | ~2M (추정 25%) |
| 활성 user | 623,866 | 200k-300k 추정 |
| Cold-start | 14,391 | 340k-440k 추정 |
| Item vocab | 29,413 | 25k-28k 추정 |

### 변경점

- `data_prep.py`: default `--last-days 28`, output `./data/cy_commerce_4w/`
- `config.yaml`: `dataset: cy_commerce_4w`, `run_name: exp_002b_bsarec_4w`
- 그 외 (`train.py`, `inference.py`, `fra.py`, `bsarec_model.py`): exp_002 와 동일 — fair ablation

### 실행

```bash
cd experiments/exp_002b_bsarec_4w
python data_prep.py                                  # ~1min
nohup python train.py > train.log 2>&1 & disown      # ~30-60min (4m 의 1/4)
nohup python inference.py > inference.log 2>&1 & disown   # ~5min
```

### 결과

학습: epoch 71 best @ RecBole LOO NDCG 0.2658, ~46분 (54 epoch 학습 후 plateau, 수동 종료).

| 모델 | self-val NDCG@10 | recall@10 | Δ vs 4m |
|---|---:|---:|---:|
| exp_002 (4m) | 0.2391 | 0.3195 | — |
| **exp_002b (4w)** | **0.2414** | **0.3242** | **+0.0023 / +0.0047** |

### 4-way sub-val 분석 (eval_val_slices.py)

| 모델 | full NDCG | no_spike NDCG | spike_only NDCG | n_no_spike |
|---|---:|---:|---:|---:|
| ALS 4m | 0.1838 | 0.0000 | 0.1846 | 4 |
| EASE | 0.1848 | 0.0833 | 0.1852 | 4 |
| BSARec 4m | 0.2391 | 0.0000 | 0.2401 | 4 |
| BSARec 4w | 0.2414 | 0.0000 | 0.2424 | 4 |

**중대 발견**: val 1,223 purchase 중 **Feb 23-26 (no_spike) 에 단 4건 (0.3%)**, Feb 27-29 spike 에 1,219건 (99.7%).

→ 사용자가 말한 "마지막 3일에 구매 70%" 는 전체 120일 기준. **val 7일 윈도우 안에서는 spike 가 99.7%**.
→ no_spike 슬라이스 n=4 통계 무의미. `full ≈ spike_only`. 분리 진단 불가능한 데이터 구조.
→ calibration ratio 2.32 의 진짜 의미: **모델의 spike 예측 능력이 normal week (Mar 1-7) 로 transfer 안 됨**. self-val 은 사실상 spike 예측 skill 만 측정.

### 결론 (제출 후 갱신 2026-05-21)

- **4w recency 효과 진짜였음** — public 0.0955 vs 4m 0.0943 = **+0.0012 (+1.3%)**. 작지만 노이즈 아님 (calibration 예측 0.0953 과 0.0002 오차 일치).
- **calibration ratio 2.53 안정** — 4m, 4w 모두 거의 동일 (2.535 vs 2.528). recency 가 calibration 자체는 안 바꿈. self-val 인플레이션의 원인은 val 의 spike-dominance.
- BSARec 의 큰 lift 는 sequential 구조 (ALS/EASE 대비 +30%). recency 는 작은 추가 lift.
- portfolio talking point 강화:
  1. **"self-val Δ 0.0023 → public Δ 0.0012 예측 정확. calibration framework 검증"** — 면접 강력
  2. "val 99.7% 가 spike — traditional sub-val 진단 불가, distribution shift 직접 진단 어려운 데이터"
  3. "sequential 모델에서도 recency 효과는 sparse 효과 trade-off 안에서만 작게 잡힘"

### 다음 액션

- exp_002c 2w / exp_002d 1w: scaffold 유지, 실행은 ROI 따라 결정 (4w 가 작긴 해도 lift 있어서 한 번 더 짧게 실험할 가치 있음)
- exp_003 DiffRec — matrix-based paradigm 추가
- **4w 의 full-data retrain (val 포함)** — 가장 유망한 final submission 후보 (Feb 1-29 학습, 0.0955 + spike 추가 boost 기대)

---

## exp_003_diffrec — QUEUED

[code: ./exp_003_diffrec/](./exp_003_diffrec/) · 출처: [references §1 DiffRec + §2 RecBole](../docs/references.md) · **SIGIR 2023**

### 가설

4번째 paradigm — **generative (diffusion)**. CF/sequential 과 family-diverse 한 ensemble 후보.

- Generative 모델은 user "intent distribution" 자체를 학습 → discriminative scoring (ALS/EASE/BSARec) 과 다른 시그널
- 4-paradigm RRF → ensemble_v1 negative 뒤집기 후보

### 알고리즘 요약

- 학습: x_0 (user binary interaction vector, dim=29.5k) 에 단계별 Gaussian noise → x_t. DNN 이 x_t → x_0 복원 학습
- 추론: 현재 x_0 → noise → reverse denoise → predicted x_0 가 item 별 score
- **item embedding 없음** — interaction vector 자체가 input/output. DNN: `[29.5k → dims_dnn → 29.5k] + timestep embedding`

전체 수식은 [논문](https://arxiv.org/abs/2304.04971) §3.

### 하이퍼파라미터 ([config.yaml](./exp_003_diffrec/config.yaml))

| 항목 | 값 | 근거 |
|---|---|---|
| `steps` (T) | 5 | 논문/RecBole default. recsys 는 small T 충분 |
| `noise_scale` / schedule | 0.001 / linear | beta 스케일링 |
| `mean_type` | x0 | x_0 직접 예측 (vs noise epsilon) |
| `dims_dnn` / `embedding_size` | [300] / 10 | small MLP / timestep emb |
| batch / lr | 2048 / 0.001 | item embedding 없어서 lr 보수적 |

모델 ≈ 17M params (item-wise dense 곱셈). BSARec (2M) 보다 큼.

### 실행

```bash
cd experiments/exp_003_diffrec
python data_prep.py                                  # 1회: ./data/cy_commerce/*.inter
nohup python train.py > train.log 2>&1 & disown      # ~2.5-3h
nohup python inference.py > inference.log 2>&1 & disown   # ~10min
```

### 핵심 질문 (결과 후 답할 것)

1. 단독으로 EASE/BSARec 수준?
2. 4-paradigm RRF (ensemble_v2) 가 ensemble_v1 negative 뒤집는가?
3. T-DiffRec (`time-aware: true`) ablation 의 lift?

### 다음 액션

- 결과 OK → `ensemble_v2_4paradigm/` (RRF + 가중치 ablation)
- LightGCN (Day 6) 진입
- Week 1 마지막: ensemble 정리 + Week 2 service 준비

---

## 전략 pivot (2026-05-21) — 모델 lift → system + LLM

데이터 특성 (99.78% view dominant + Feb 27-29 spike + self-val/public ratio 2.32) 때문에 NDCG lift 짜내기 ROI 가 매우 낮음. portfolio 무게중심을 다음과 같이 이동:

| Was | Now |
|---|---|
| 5 모델 + ensemble lift 추구 | **Week 2 FastAPI 멀티 모델 서빙 + Solar API conversational layer** |
| 4-week / hyperparam ablation | paradigm coverage 만 (light) |
| mini-L3Rec 검토 | skip — lift 가능성 낮고 시간 부담 |

**Week 1 남은 진행**:
- exp_002 (BSARec 4m) — 끝까지 (학습 중)
- exp_002b (BSARec 4w) — queued, **결과 기록 후 light wrap-up**
- exp_003 (DiffRec) — paradigm coverage 차원에서만, lift 추구 X
- LightGCN — Day 6 paradigm coverage 차원에서만
- ensemble v2 (5-model RRF) — 한 번만 정리용

**Skip 결정**:
- mini-L3Rec (A/A')
- BSARec α/c hyperparam sweep
- 추가 ablation, longer training, fine-tuning

**Week 2 메인** (포트폴리오 가치 최대):
- FastAPI 멀티 모델 서빙 (`predict_for_user(user_id, top_k, model='auto'|'ease'|...)`)
- Latency SLO + fallback chain (CF → popularity → empty)
- Prometheus 메트릭 + structured logging
- **Solar API conversational layer** — query → intent → CF retrieval → 자연어 응답

## 컨벤션 — 새 실험 추가 시

1. **폴더 생성**: `experiments/exp_NNN_<name>/` 에 코드 + config 만. README 안 만듦
2. **이 파일 (`experiments/log.md`) 에 섹션 추가**: 가설 / 하이퍼 / 실행 / 결과 / 결론. 결과 부분은 TBD 로 두고 학습 후 채움
3. **leaderboard + 제출 이력 표 업데이트**
4. **`docs/references.md`** 에 새 모델/라이브러리 row 추가
