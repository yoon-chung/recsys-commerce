# EDA Findings — Commerce Purchase Prediction

---

## 1. 데이터 스케일

| 항목 | 값 |
|---|---:|
| Unique users | 638,257 |
| Unique items | 29,502 |
| Sessions | 2,889,552 |
| Total interactions | 8,350,311 |
| Sparsity | 99.96% |
| 기간 | 2019-11-01 ~ 2020-02-29 (120일) |
| 제출 row 수 | 6,382,570 (= 638,257 × 10) |
| 제출 user ⊆ train user | ✅ (cold-start 신규 ID 없음) |

**스키마 제약 — user demographic feature 없음**

CLAUDE.md 스키마: `user_id, item_id, user_session, event_time, category_code, brand, price, event_type`. user-side feature는 `user_id` 외 0개 (연령/성별/지역 등 미제공).

**모델링 함의**:
- user 표현은 100% **behavioral derivation** (history → aggregate features)
- demographic-based segmentation 불가 → collaborative filtering 시그널 의존도 매우 높음 (ALS / SASRec 가 효과 큰 구조적 이유)
- Two-stage reranker stage 2의 user-side feature도 **behavioral aggregate만** 가능 (top-brand, top-category, 평균 price, session 패턴 등 — history에서 derive)
- cold-start fallback이 popularity로 한정되는 이유 — user-side를 보조할 demographic이 없음

---

## 2. Event 분포 (★ 가장 충격적 발견)

| event_type | 비율 | 절대 건수 |
|---|---:|---:|
| view | **99.78%** | 8,331,873 |
| cart | 0.20% | 16,362 |
| purchase | **0.02%** | 2,076 |

> Train 전체 4개월에 purchase가 단 **2,076건**. 이 중 69.2%가 마지막 3일(Feb 27~29)에 몰려있음 (섹션 5 참조).

**시사점**:
- 모델은 view-dominated 데이터에서 학습됨 → purchase 예측은 본질적으로 view→purchase로 가는 약한 시그널을 잡아내는 task
- event 가중치를 어떻게 설계하느냐가 큰 영향 (e.g. view=1/cart=3/purchase=5 vs 모두 1)
- baseline ALS는 모두 1 (label=1 + groupby sum)을 씀 → view 다중 방문이 자연 가중치 역할

---

## 3. 세션 구조

| 항목 | 값 |
|---|---:|
| 평균 세션 내 view 비중 | 99.55% |
| view-dominant 세션 | 2,876,069 / 2,889,552 (**99.53%**) |
| cart-dominant 세션 | 11,686 (0.40%) |
| purchase-dominant 세션 | 1,797 (0.06%) |
| 단일 event_type만 있는 세션 | 2,882,658 (**99.76%**) |

**시사점**: 세션 단위로 보면 거의 **모드 분리** (browsing 세션 vs shopping 세션). 시퀀셜 모델 입장에선 세션 경계를 어떻게 다룰지 (intra-session sequence vs inter-session aggregate) 설계 포인트.

---

## 4. User / Item 분포 (롱테일)

### User-side
```
유저당 인터랙션 수 분포
  count       638,257
  mean             13.08
  std              58.80
  min               1
  25%               2
  50% (median)      6
  75%              15
  90%              29
  95%              45
  99%             102
  max          37,207
```

| 그룹 | 비율 |
|---|---:|
| 인터랙션 1건 user | 16.8% |
| 인터랙션 ≤5건 user | 47.5% |
| 인터랙션 >50건 user | **4.1%** ← 롱시퀀스 유저는 소수 |

### Item-side
```
아이템당 인터랙션 수 분포
  count        29,502
  mean            283
  std           1,262
  min               1
  25%              30
  50% (median)     77
  75%             221
  90%             579
  95%           1,045
  99%           3,347
  max         143,759
```

| 커버리지 | 비율 |
|---|---:|
| 상위 1% 아이템이 차지하는 인터랙션 | 25.1% |
| 상위 10% 아이템이 차지하는 인터랙션 | 63.4% |
| 1회만 등장한 아이템 | 0.1% |

**시사점**:
- **롱시퀀스 유저 4.1%만** — Mamba4Rec처럼 long-seq 처리 우월성 강조하는 모델 우선순위 낮음
- **moderate longtail** — 상위 10%가 63% 차지. popularity baseline이 어느 정도 base 점수 줄 수 있음

---

## 5. 🚨 Feb 27~29 Purchase Spike (가장 중요한 분포 이슈)

전체 4개월 중 **마지막 3일에 purchase가 폭증**:

| 날짜 | view | cart | purchase | 평소 대비 |
|---|---:|---:|---:|---|
| 2/26 (직전) | 63,007 | 40 | 1 | (직전 일평균 ~4) |
| **2/27** | 6,533 | 0 | **668** | **167x** |
| **2/28** | 33,819 | 15 | **410** | 100x |
| **2/29** | 39,383 | 20 | **359** | 90x |

| 비교 | 값 |
|---|---:|
| 2월 1~26일 일평균 purchase | ~1.3 |
| Feb 27~29 일평균 purchase | 479 |
| **Feb 27~29 / 전체 4개월 purchase** | **69.2%** |

**무엇을 의미하는가**:
- Train 전체에서 purchase 시그널의 약 70%가 이 3일에 집중
- 우리 자체 validation (마지막 7일 = 2/23~2/29)의 ground truth = 1,443 purchase 중 거의 모두가 이 3일에서 나옴
- 평소 1일 4건 → 갑자기 668건. 일반적인 e-commerce 패턴 아님 (월말 캠페인? 데이터 anomaly? 추후 검증 필요)
- **자체 val NDCG@10이 public NDCG@10과 크게 다를 수 있다** — public test (Mar 1~7)가 평소 분포라면 spike와 다른 패턴이고, public이 spike 직후 효과를 반영하면 또 다름

**실험 설계에 미치는 영향**:
- 자체 val 점수만 보고 모델 우열 판단하면 위험
- public 제출 점수와 함께 봐야 함 — 제출 비용 ↑
- spike를 별도 분석할 가치 있음: 어떤 카테고리/브랜드/유저층에서 발생했는지

---

## 6. 행동 전환 (Conversion Funnel)

같은 (user, item) 쌍에서 시간순으로 발생한 전환:

| 전환 타입 | 건수 | from 비율 |
|---|---:|---:|
| view → cart | 45,665 | 0.548% of view rows |
| view → purchase | 6,116 | 0.073% of view rows |
| cart → purchase | 130 | 0.795% of cart rows |

| 첫 발생 시각 기준 | 값 |
|---|---:|
| view-있는 (user,item) 쌍 | 5,936,433 |
| view 후 cart 진행 | 6,354 (0.11%) |
| view 후 purchase 진행 | 1,035 (0.02%) |
| cart 후 purchase 진행 | 59 |

**시사점**:
- view → purchase 전환율이 극단적으로 낮음 (0.073%) — view가 강한 구매 의도 신호는 아님
- **cart는 view보다 10배 강한 신호** (cart→purchase 0.795% vs view→purchase 0.073%) — cart 발생을 우대하는 weighting 정당화 가능. 다만 cart event 자체가 희귀 (16k 건)
- ALS confidence 가중치 설계: cart > purchase > view 순서가 합리적일 수도? (purchase는 이미 발생한 것이라 추천 의미 없음 — purchase 가중치를 0으로 둘 수도)

### 아이템 단위 차이
| 그룹 | item 수 | 가격 중앙값 | view 중앙값/item |
|---|---:|---:|---:|
| cart 경험 있는 item | 3,854 | 67.70 | **497** |
| view-only item | 25,648 | 62.81 | **62** |

cart 경험 item은 view가 평균 8배 많이 발생함 → cart는 인기·의도 모두 강한 시그널

### cart 경험 상품의 cat_l2 상위
shoes(2441) > scarf(322) > shirt(227) > costume(223) > underwear(195) > shorts(97) > trousers(80) > tshirt(76) > jeans(69) > pajamas(3)

**의류 카테고리에 cart가 집중** — fashion 관련 추천 휴리스틱이 유효할 수 있음

---

## 7. 아이템 반복 패턴

(user 입장에서 같은 item을 반복 인터랙트하는 비율)

| 항목 | 값 |
|---|---:|
| 평균 반복도 (전 이벤트 기준) | **14.63%** |
| 반복도 > 0 user | 47.3% |
| 반복도 > 50% user | 6.5% |

**시사점**:
- 14.63% > 10% 구간 → **NBR (Next Basket Recommendation) 계열 시그널 유효**
- 모델 권고: **TIFU-KNN을 앙상블 보조 멤버로** 검토 가치
- SAFERec 같은 반복 빈도 모델은 후보군에 두기 (단, 보조)

---

## 8. 시간 간격 (Time Gap)

| 항목 | 값 |
|---|---:|
| 연속 이벤트 간격 CV | **4.12** |

**시사점**:
- CV가 매우 큼 (정상분포 기준 1 이상이면 high variance) → 단순 sequential은 시간 정보 손실
- 모델 권고: **TiSASRec** (Time-interval-aware SASRec) 강력 추천

---

## 9. 시퀀스 길이 (학습 구간 기준)

학습 구간(2019-11-01 ~ 2020-02-22) user 시퀀스 길이 percentile:

| Percentile | 시퀀스 길이 | User 커버리지 |
|---|---:|---:|
| p50 | 6 | 52.7% |
| p75 | 15 | 76.5% |
| **p90** | **29** | 90.3% |
| p95 | 44 | 95.1% |
| p99 | 100 | 99.0% |

**권고 MAX_ITEM_LIST_LENGTH = 29~32** (p90 기준, 20~200 범위에서 clip).

p90 = 29인데 RecBole 기본 sasrec.yaml은 50. **50을 그대로 두면 90% user는 padding으로 채워짐** → 학습 효율 손해, MAX=32 정도가 합리적.

---

## 10. Validation Split User 커버리지

| 그룹 | 값 |
|---|---:|
| 학습 구간 (~2020-02-22) user | 623,866 |
| 검증 구간 (마지막 7일) user | 61,310 |
| 전체 user (= 제출 user) | 638,257 |
| **검증 구간 미등장 user** | **576,947 (90.4%)** ← popularity fallback 적용 |
| 검증·학습 모두 등장 (eval target) | ~60,000 |
| eval target ∩ purchase 있는 user (실제 NDCG 측정 대상) | **928** ([exp_000](../experiments/exp_000_als_baseline/) 측정값) |

**시사점**:
- 자체 val NDCG 계산은 **928명만**에서 평균. 분산이 큼
- 1주일 hold-out으로는 통계적 안정성 부족할 수 있음 → val_days=14 늘리기 고려 가능 (대신 train 1주 잃음)

---

## 11. 모델 선택 권고 (팀원 EDA + 우리 해석)

| 모델 | 추천도 | 근거 |
|---|---|---|
| **TiSASRec** | ★★★ | time gap CV=4.12 (high) → 시간 간격 직접 모델링 필수 |
| **BSARec** (AAAI 2024) | ★★★ | 최신 sequential, NDCG +10~14% literature, RecBole 호환 |
| **FEARec** | ★★★ | RecBole 즉시 실험 가능 |
| **SAFERec** | ★★★ | item 반복도 14.6% > 10% → 빈도 신호 작동 |
| **TIFU-KNN** | ★★☆ | 앙상블 보조 멤버. 자체로는 약하지만 다양성 추가 |
| ALS / EASE / LightGCN | ★★☆ | candidate generation 또는 reranker 입력. 단독으로는 sequential 대비 약함 (예상) |
| **Mamba4Rec** | ★☆☆ | 롱시퀀스 4.1%만 → 효율 우월성 발현 어려움 |
| CL4SRec | ★☆☆ | contrastive learning, 노이즈 대처는 좋지만 우리 데이터 노이즈가 그렇게 크지 않음 |

---

## 12. 초기 앙상블 가중치 (팀원 권고)

EDA 셀 47 출력 기준:

| Model | Weight |
|---|---:|
| TiSASRec | 0.35 |
| BSARec | 0.30 |
| SAFERec | 0.20 |
| TIFU-KNN | 0.15 |
| CL4SRec | 0.00 |

합 = 1.00. 합 = 1이 되도록 SAFERec과 TIFU-KNN을 반복도(14.6%) 임계치로 조정한 결과. 추후 실제 self-val/public 점수가 나오면 weighted average 또는 Reciprocal Rank Fusion (RRF)로 재조정.

---

## 13. 미해결 질문 (1차 EDA)

EDA에서 답하지 못한 것 — **2026-05-21 확장 분석에서 대부분 해결, §16 참조**:

1. **Feb 27~29 spike의 정체** → §15.1 (apparel.shoes 폭증, IT 브랜드, 90.5%가 기존 user)
2. **price 분포의 모델 활용** → §15.2 (purchase는 user 평소 price band 안 86.8%)
3. **session 경계 정의** → §15.3 (median 21시간 gap, view→purchase 94.6%가 inter-session)
4. **brand affinity** → §15.4 (view top brand ↔ purchase top brand 일치율 46.5%)
5. **요일/시간대 패턴** → §15.5 (spike 빼면 시그널 거의 없음, cyclic encoding 가치 낮음)

남은 질문 (실험·제출 후에 답 가능):
- public test (Mar 1~7)도 spike의 연장선상인가?

---

## 14. 실험 계획 영향 (실제로 어떻게 사용할 것인가)

각 실험 README에서 이 문서의 관련 섹션을 링크하면 의사결정 근거 명확:

- [exp_000_als_baseline](../experiments/exp_000_als_baseline/) → 섹션 2 (event 분포) + 섹션 10 (validation coverage). 자체 val NDCG가 public과 크게 다르다면 섹션 5의 spike가 원인일 가능성 우선 검토.
- exp_002 (sequential 진입 시) → 섹션 8 (TiSASRec) + 섹션 9 (MAX_LEN=29~32) + 섹션 11 (모델 추천)
- exp_003 (NBR/반복 빈도) → 섹션 7 (반복도) + 섹션 11 (SAFERec/TIFU-KNN)
- ensemble exp → 섹션 12 (초기 가중치)

---

## 15. 확장 EDA — 미해결 질문 분석 (2026-05-21)

스크립트: `baseline/data/eda_extended.py` (gitignored). §13의 미해결 질문 5개를 데이터로 분석.

### 15.1 Feb 27~29 Spike의 정체

**카테고리 (l1 / l2)**:
- spike 1,437건 모두 **apparel 카테고리 (100%)** — l1 단일
- l2: **apparel.shoes 1,010건 (70%)**, trousers 136, shirt 63, costume 59, scarf 40 — 의류 전반
- base period (Feb 1~26) 148건도 거의 apparel.shoes (106건). spike는 같은 카테고리의 **양적 폭증**

**Brand — 이상 신호**:

| brand | spike | base |
|---|---:|---:|
| xiaomi | 171 | 21 |
| sony | 122 | 9 |
| iqos | 96 | 4 |
| samsung | 81 | 7 |
| apple | 45 | - |

⚠️ **데이터 anomaly 의심**: 카테고리는 apparel(의류)인데 brand는 IT/가전 (xiaomi/sony/iqos/samsung/apple) — 일반 commerce 데이터에서 자연스럽지 않음. 가능성: (a) brand-category 매핑 오류, (b) IT 브랜드의 어패럴 라이센스 라인, (c) raw 데이터셋 자체의 라벨 노이즈. **모델 학습에는 둘 다 그대로 입력해서 학습 (text feature 아니라 ID embedding)** → 안전.

**구매자 분포**:
- spike unique 구매자: **1,100명**
- spike 이전에 view 이력 있던 user: **995명 (90.5%)**
- spike 신규 user (이력 0): 102명 (9.3%)
- pre-spike view 횟수: median **14건**, mean 29 (활발히 둘러보던 user들)

**모델링 함의**:
- spike 구매자의 90%는 **사전 view 이력이 풍부한 기존 user** → cold-start 문제 아님
- sequential 모델 (SASRec / TiSASRec) 이 사전 view를 보고 다음 purchase 예측하는 데 적합한 시나리오
- TIFU-KNN 같은 NBR 모델도 candidate generation 단계에 유용 (view 이력 → re-purchase intent)

### 15.2 Price 분포의 모델 활용

**event_type별 price (median, USD)**:

| event | median | mean | p90 |
|---|---:|---:|---:|
| view | $79 | $151 | $384 |
| cart | $66 | $149 | $387 |
| purchase | $64 | $122 | $384 |

**purchased item vs view-only item median price**:
- purchased item median $64 vs view-only item median $63 — 거의 동일
- → **price 자체는 item-level purchase 여부를 강하게 결정하지 않음**

**user-level price band — ★ 강한 시그널**:
- view ≥5건 + has purchase user 1,397명 분석:
- purchase price가 user의 view IQR(p25~p75) 안: **55.5%**
- view median ± 1.5×IQR 안: **86.8%** ← reranker feature로 충분히 활용 가치
- ratio (purch median / view median) = **1.0 (median)** — user는 자기 평소 가격대에서 구매

**모델링 함의**:
- **Two-stage reranker 핵심 feature**: "candidate price ∈ user view price band 안인가" 가 강한 시그널
- candidate price ÷ user view median 비율, log-distance 등 derive 가능
- item-side price feature보다 **user-item interaction** 형태가 강함

### 15.3 Session 경계 정의

**인접 session 시간 gap 분포** (median, 분):
- median: **1,279분 ≈ 21시간** → session은 약 **하루 단위로 갈림**
- 5분 이내 새 session도 355k 건 (정의가 시간만 보는 게 아닌 듯 — logout/device 변경 등)

**Session 내 event 구성** (전체 2,889,552 sessions):
- view 포함: 2,878,795
- cart 포함: 15,695 (0.5%)
- purchase 포함: 1,980 (0.07%)
- 3-type 모두: 24 (희귀)

**view → purchase 전환 — 가장 중요한 발견**:
- view 후 purchase가 일어난 (user, item) 쌍 1,198건 중:
- **같은 session 내 view→purchase: 5.4% (65건)**
- **다른 session에서 purchase: 94.6% (1,133건)**

**모델링 함의** (큰 의사결정에 영향):
- view ↔ purchase는 **거의 항상 다른 session**에서 일어남
- → **session-level 모델 (SR-GNN, session-aware NARM 등) 우선순위 ↓**
- → **user-level long-term sequence 모델 (SASRec, TiSASRec, MB-STR) 우선순위 ↑** — 이미 우리 선택과 일치
- session 경계를 sequence 학습 시 "필수 boundary"로 다루지 않아도 됨. user의 전체 cross-session sequence를 그대로 학습

### 15.4 Brand Affinity

**brand 인터랙션 반복도** (user × brand 단위):
- (user, brand) pair: 3.1M
- multi-interact (n≥2): **39.5%** — user는 같은 brand에 반복 노출됨
- user별 unique brand 수: median 3, p90 11

**brand 반복 구매 — 매우 약함**:
- 같은 user × 같은 brand 2회+ 구매: 178 pairs (대부분 spike 때문)
- base period 한정: 23 pairs only (n=639 purchase 중)
- → purchase 자체가 너무 sparse해서 **brand 단위 NBR 시그널은 약함**

**view top brand ↔ purchase top brand 일치율 — ★ 강한 시그널**:
- **46.5%** (1,597명 중 743명)
- 우연 baseline (1 / median 3 brand) ≈ 33%보다 강함
- → **user가 가장 많이 본 brand가 가장 많이 사는 brand에 강한 양의 상관**

**모델링 함의**:
- TIFU-KNN을 brand 단위로 확장하는 건 데이터 sparsity상 어려움 (brand 단위 purchase가 너무 적음)
- **Two-stage reranker feature로**: "candidate brand == user의 view top brand?" (binary or rank-1 indicator) — 강력
- Brand embedding을 SASRec input feature에 추가하는 것도 ablation 가치 (item embedding과 concat)

### 15.5 요일/시간대 패턴

**요일별 purchase**:

| dow | purchase | purchase_rate |
|---|---:|---:|
| Mon | 67 | 0.00006 |
| Tue | 60 | 0.00005 |
| Wed | 80 | 0.00007 |
| **Thu** | **792** | **0.00072** ← spike 시작 (2/27=Thu) |
| **Fri** | **505** | **0.00042** ← spike (2/28=Fri) |
| **Sat** | **443** | **0.00035** ← spike (2/29=Sat) |
| Sun | 129 | 0.00010 |

**Spike 제외 시 (Feb 27~29 빼고)**:

| dow | purchase | purchase_rate |
|---|---:|---:|
| Mon | 67 | 0.00006 |
| Tue | 60 | 0.00005 |
| Wed | 80 | 0.00007 |
| Thu | 124 | 0.00011 |
| Fri | 95 | 0.00008 |
| Sat | 84 | 0.00007 |
| Sun | 129 | 0.00010 |

→ **spike 제외 후 요일 효과 거의 없음** (Thu만 약간 높지만 ~2배 미만). 요일 cyclic encoding 가치 미미.

**시간대별 purchase rate (UTC)**:
- UTC 3~13시: 0.0003 (purchase rate 높음, view 활성 시간)
- UTC 18~20시: 0.00013 (view 많지만 purchase 적음)
- → 시간대별 차이는 있지만 spike에 의해 강하게 distort됨

**모델링 함의**:
- 요일/시간대 cyclic encoding의 lift 기대치 **낮음** — spike 효과로 시그널이 distort되어 generalization에 도움 안 될 가능성
- TiSASRec의 time-interval embedding (event 간 간격)이 **요일/시간대 absolute time보다 더 강한 시그널**일 가능성 ← 우선순위 유지
- Two-stage reranker에 요일/시간대 추가 검토는 **하후순위**. price band / brand affinity 가 더 강력한 feature

### 15.6 요약 — 모델링 우선순위 업데이트

기존 결론 (§11) + 확장 EDA 결과를 종합:

| 시그널 | 강도 | 활용 |
|---|---|---|
| **사전 view 이력 → next purchase** (90.5%) | ★★★ | SASRec / TiSASRec / MB-STR |
| **user-level price band** | ★★★ | Two-stage reranker feature |
| **view top brand → purchase top brand** (46.5%) | ★★★ | Two-stage reranker feature, brand embedding |
| **cross-session sequence** (purchase 94.6% inter-session) | ★★★ | user-level sequential 모델 (session-aware보다) |
| **item 반복도 (14.6%)** | ★★ | NBR (TIFU-KNN), `filter_already_liked=False` |
| **time interval irregularity** (CV=4.12) | ★★ | TiSASRec |
| **요일/시간대 cyclic** | ★ | 후순위, lift 제한적 |
| **session 경계** | ☆ | session-level 모델 우선순위 낮춤 |

**의사결정 변경**:
- §11에서 ★★☆였던 LightGCN/EASE는 그대로 유지 (CF 시그널)
- §11에서 다루지 않았던 **Two-stage reranker가 매우 강력해질 잠재력** — price band + brand affinity 두 강력한 feature 확보
- session-aware 모델 (SR-GNN, NARM session 모드) 우선순위 **하향** — cross-session pattern이 dominant

---

## 16. 데이터 보안 노트

이 문서는 commit 가능. raw user_id / item_id, 데이터 파생물 (user2idx.json 등) 미포함. 모두 집계 통계와 분석 결과. 만약 추가 분석에서 raw id가 포함된 표 등을 만들면 gitignored 영역(예: `experiments/local_notes/`)에 둘 것.

원본 `core/eda.ipynb` 자체는 셀 출력에 raw id가 박혀있을 수 있어 `.gitignore`로 보호 중. 신규 노트북도 동일 패턴 (`*.ipynb`) 적용됨. 확장 분석 스크립트 `baseline/data/eda_extended.py`도 `baseline/` 경로 자체가 `.gitignore`로 보호되어 raw id 노출 위험 없음 (집계 결과만 docs/eda_findings.md에 옮겨 commit).

---

## 17. 멘토링 질문 메모 (2026-05-21)

cy 본인 참고용. 멘토링 종료 후 정리/삭제 가능. EDA 분석 결과를 근거로 묶음.

### Q1. 데이터 anomaly — apparel × IT brand 매핑 정상인가?

**EDA 근거** (§15.1): Feb 27-29 spike 1,437건 모두 `apparel.shoes` 등 의류 카테고리. 그런데 brand는 xiaomi (171), sony (122), iqos (96), samsung (81), apple (45) 등 모두 IT/가전 브랜드. base period (Feb 1-26)에서도 동일 패턴.

**물어볼 것**:
- 데이터 자체의 brand-category 정합성 이슈인가, 아니면 IT 브랜드의 의류 라이센스 라인이 실제로 있는 건가?
- ID embedding 기반 모델 (SASRec 등) 에는 영향 없을 듯하지만, brand text feature를 활용하는 시도(예: brand 임베딩을 워드 임베딩으로 init)를 한다면 주의해야 하는지?

### Q2. Time Series CV 전략 (팀원 질문)

**EDA 근거**: purchase 2,076건 중 1,437건(69%)이 Feb 27-29 3일에 집중. 우리 self-val/public ratio 2.32 중 약 절반이 이 spike 효과로 추정.

**물어볼 것**:
- rare-event burst 데이터에서 K-fold CV가 의미가 있나? (각 fold의 분포가 너무 달라짐)
- 현재 time-based holdout (last 7 days, eval user 928명) → Feb 9-22로 hold-out 옮기면 spike를 train에 두고 평가 안정성 ↑. 멘토 권고?
- nested CV / purged CV / walk-forward 중 우리 task에 가장 정합한 방식?

### Q3. 역대 기수 이 task 점수 분포

**상황**: 우리 exp_000 ALS = self-val 0.184 / public 0.079. baseline 공시 0.0847.

**물어볼 것**:
- 역대 기수 이 commerce purchase task 상위권 (1~3위) 점수가 어느 정도인가? (대략 범위라도)
- calibration ratio (self-val ÷ public) 2.32가 이 task에서 일반적인 수준인가, 아니면 우리 self-val window가 spike에 너무 노출되어 inflated된 건가?

### Q4. mid-project Hydra refactor 의견

**상황**: 2주 남음. 팀원은 Hydra 기반 repo (`github.com/WanYoung-Oh/recsys`), 우리는 자체 config.yaml + load_config 패턴. cy는 Hydra 학습 욕심도 있음.

**물어볼 것**:
- 멘토 본인 경험상 mid-project Hydra 도입 ROI는 어떤가? 어느 정도 규모 프로젝트부터 가치 있는가?
- 우리 case (단독 작업, 2주, 적극적 튜닝 필요) → wandb sweeps / Optuna 직접이 충분한가?
- 대회 후 portfolio refactor 프로젝트로 Hydra 학습이 더 적절한지?

### Q5. 팀 협업 — 모델 분업 vs 같은 모델 재현

**상황**: 팀원이 8개 sequential 모델 구현 중 (sasrec/tisasrec/bsarec/fearec/mbstr/saferec/tifu_knn/cl4srec). 우리는 §15.6 결론대로 Two-stage reranker가 강력 잠재력. EASE 같은 다양성 멤버도 고려.

**물어볼 것**:
- 같은 모델을 두 명이 다른 hyperparam으로 돌려 ensemble 다양성 확보 vs 모델 family 분담 — 어느 쪽이 RRF lift 큰가?
- RRF vs 가중평균 vs stacking — 우리 sparse purchase GT 데이터에 가장 정합한 방법?
- 본인 portfolio도 고려할 때 sequential 깊게 vs 다양성 확보 — 어느 쪽 추천?

### Q6 (자투리, 시간 남으면). EDA에서 새로 발견한 시그널 확인

**EDA 근거** (§15.6 요약):
- user-level price band: purchase의 86.8%가 user view band 안 → reranker feature로 강력?
- view top brand ↔ purchase top brand 일치율 46.5% → 우연 33% 대비 강함
- view → purchase 94.6%가 inter-session → user-level long-term sequence 모델 우선

**물어볼 것**:
- price band, brand top-1 일치 같은 feature가 commerce 실무에서도 자주 쓰이는 strong signal인가?
- cross-session vs intra-session 패턴은 멘토 경험상 어떤 모델군이 강한가?
