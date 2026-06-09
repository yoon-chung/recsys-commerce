# EDA → 모델 선택 매핑 (신호 × 모델 매트릭스)

**목적**: "왜 이 모델 조합인가"를 한 장으로 정당화.
EDA 수치([eda_findings.md](eda_findings.md)) → 신호 → 모델 커버리지 → 최종 2-stage 파이프라인의 인과를 압축.

**상태**: 사후 정리본 (실험·제출 결과 반영).

---

## 1. EDA 핵심 수치 → 모델 선택 함의

| 지표 | 값 | 출처 | 모델 선택 함의 |
|---|---|---|---|
| 연속 이벤트 간격 CV | **4.12** | §8 | 시간 간격 명시 모델 필수 → TiSASRec |
| 시퀀스 길이 median / p90 | 6 / **29** | §9 | 짧은 시퀀스 위주 → 노이즈 필터(BSARec) 유효, 롱시퀀스 모델 불필요 |
| 롱시퀀스(>50) 유저 | **4.1%** | §4 | Mamba4Rec 등 long-seq 우월성 발현 어려움 → 제외 |
| 아이템 반복도(전 이벤트) | **14.6%** | §7 | >10% → NBR/빈도 신호 유효 → TIFU-KNN |
| 이벤트 타입 | view/cart/purchase | §2 | 다중 행동 신호 존재 → MB-STR (행동 타입 임베딩) |
| 상위 1% 아이템 커버리지 | **25.1%** | §4 | moderate longtail → 인기 편향 교정 멤버 필요 → TIFU-KNN |
| cross-session purchase | **94.6%** inter-session | §15.3 | user-level 장기 시퀀스 ≫ session-level → SR-GNN/NARM 제외 |
| view→purchase 전환 user | **90.5%** 사전 view 보유 | §15.1 | cold-start 아님 → 시퀀스 모델이 사전 view로 예측 가능 |

---

## 2. 신호 × 모델 커버리지 매트릭스

> ✓✓ = 핵심 강점 · ✓ = 부분 커버 · ✗ = 미포함

| 신호 (EDA 근거) | TiSASRec | BSARec | MB-STR | TIFU-KNN | CL4SRec | SAFERec | Mamba4Rec |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 시간 간격 모델링 (CV=4.12) | ✓✓ | ✗ | ✗ | ✓ (감쇠) | ✗ | ✗ | ✗ |
| FFT 저역통과 노이즈 제거 (median=6) | ✗ | ✓✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 행동 타입 임베딩 (view/cart/purchase) | ✗ | ✗ | ✓✓ | ✗ | ✗ | ✓ (view 빈도) | ✗ |
| 반복/빈도 NBR 신호 (14.6%) | ✗ | ✗ | ✗ | ✓✓ | ✗ | ✓ | ✗ |
| 비신경망 다양성 / longtail 교정 (상위1%=25%) | ✗ | ✗ | ✗ | ✓✓ | ✗ | ✗ | ✗ |
| 대조학습 증강 (노이즈 대응) | ✗ | ✗ | ✗ | ✗ | ✓✓ | ✗ | ✗ |
| cross-session 장기 시퀀스 (94.6%) | ✓✓ | ✓✓ | ✓✓ | ✓ | ✓ | ✓ | ✓✓ |

---

## 3. 채택한 2-stage 파이프라인

**Stage 1 (candidate generation, top-50 union) → Stage 2 (LGBM LambdaRank reranker) = Public 0.1358** ([exp_010b](../experiments/log.md))

| Stage 1 모델 | 역할 | 단독 self-val NDCG@10 |
|---|---|---|
| TIFU-KNN | 비신경망 시간 감쇠 — 반복/빈도 + longtail (이 데이터의 dominant signal) | **0.292** (single best) |
| BSARec | AAAI 2024, FFT 저역통과로 짧은 시퀀스 노이즈 정리 | 0.247 |
| MB-STR | view/cart/purchase 행동 임베딩 (이 데이터 고유 신호) | 0.239 |
| BSARec+CL | 대조학습 증강 ablation (paradigm coverage) | 0.224 |
| BERT4Rec | bidirectional MLM (paradigm coverage) | 0.216 |

> **Single TIFU = 0.1175 → 2-stage reranker = 0.1358 (+15.2%)**. 단순 RRF/가중 ensemble 4종은 모두 single best를 못 넘었음 — score-blind aggregation 한계. LGBM reranker가 LOO-overfit 모델을 자동 down-weight해서 lift가 살아남.

> **현업 관점**: 5-model + reranker는 paradigm coverage 입증이고, production 서빙에서는 latency·비용 때문에 single-model + 강한 FE/reranker가 현실적인 형태. 이 파이프라인의 lift도 reranker(+15.2%)가 ensemble(0%)을 압도 — production constraint와 같은 방향.

---

## 4. Stage-2 reranker 신호 (candidate gen과 별개)

위 매트릭스는 candidate generation(stage 1) 기준. reranker(stage 2)에서 실제로 점수를 끌어올린 신호는 EDA가 따로 예측했고, exp_010 feature importance가 확인해줌:

| reranker 신호 | EDA 예측 | 실제 feature importance (exp_010) |
|---|---|---|
| **TIFU rank/score** | §7 반복도 14.6% + §4 longtail | **1·3위** (`tifu_rank`, `tifu_norm_score`) — backbone |
| user-item recency | §8 시간 간격 CV=4.12 | **2위** (`ui_days_since_last`) |
| cart/purchase 이력 (개수) | §6 cart-bypass 96% → binary는 약할 것 | 4·5위 (`item_cart`, `user_repeat_ratio`) — 카운트 형태는 살아남 |
| cart/purchase 이력 (binary) | 동일 예측 | **importance ≈ 0** (예측 적중) |
| price band / brand affinity | §15.2/§15.4 (★★★) | **투입했으나 importance 낮음** — heavy FE 추가 여지 있으나 ROI 낮을 가능성 |

---

## 5. 한 줄 결론

> EDA 수치(반복 14.6% / 시간 간격 CV 4.12 / 행동 3종 / longtail)가 가리킨 **반복·시간·행동·다양성** 4축에 대응해 5개 base + LGBM reranker로 0.1358 달성. **데이터의 dominant signal(반복 + temporal decay)을 가장 직접 모델링한 TIFU-KNN이 single best (0.1175)였고, reranker가 거기서 +15.2% lift** — score-blind ensemble은 lift 0인 반면 score-aware reranker는 살아남는 production 패턴을 동일 dataset에서 직접 검증.

---

*집계 통계·분석 결과만 포함. raw id 미포함, commit 가능 ([eda_findings.md](eda_findings.md) §16 동일 기준).*