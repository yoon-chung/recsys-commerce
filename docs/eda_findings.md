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
| eval target ∩ purchase 있는 user (실제 NDCG 측정 대상) | **928** ([exp_000](../members/cy/exp_000_als_baseline/) 측정값) |

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

## 13. 미해결 질문 (다음 분석)

EDA에서 답하지 못한 것 — 이후 실험·재분석으로 채울 것:

1. **Feb 27~29 spike의 정체**:
   - 어떤 카테고리/브랜드에서 폭증했나? cat_l2 별 spike-vs-baseline 비교 필요
   - spike 구매자가 평소 view 패턴이 있던 user인가 vs 새 user인가?
   - public test (Mar 1~7)도 spike의 연장선상인가? — 우리는 알 수 없음, 제출해야 함
2. **price 분포의 모델 활용**:
   - purchase의 price 중앙값 vs view-only item의 price 비교 (이미 EDA 일부 다룸)
   - user 별 평소 price band → 추천 후보 필터링에 활용 가능?
3. **session 경계 정의**:
   - user_session 경계는 어떻게 정해졌나? (timeout? user logout?)
   - intra-session vs inter-session 정보를 둘 다 쓰는 hybrid 모델이 유리한가
4. **brand affinity**:
   - 같은 brand 반복 구매 비율 — TIFU-KNN의 NBR 효과를 brand 단위로 확장 가능?
5. **요일/시간대 패턴** (EDA에 일부 있음):
   - cyclic 인코딩으로 feature 추가하면 reranker가 잡아낼 수 있는지

---

## 14. 실험 계획 영향 (실제로 어떻게 사용할 것인가)

각 실험 README에서 이 문서의 관련 섹션을 링크하면 의사결정 근거 명확:

- [exp_000_als_baseline](../members/cy/exp_000_als_baseline/) → 섹션 2 (event 분포) + 섹션 10 (validation coverage). 자체 val NDCG가 public과 크게 다르다면 섹션 5의 spike가 원인일 가능성 우선 검토.
- exp_002 (sequential 진입 시) → 섹션 8 (TiSASRec) + 섹션 9 (MAX_LEN=29~32) + 섹션 11 (모델 추천)
- exp_003 (NBR/반복 빈도) → 섹션 7 (반복도) + 섹션 11 (SAFERec/TIFU-KNN)
- ensemble exp → 섹션 12 (초기 가중치)

---

## 15. 데이터 보안 노트

이 문서는 commit 가능. raw user_id / item_id, 데이터 파생물 (user2idx.json 등) 미포함. 모두 집계 통계와 분석 결과. 만약 추가 분석에서 raw id가 포함된 표 등을 만들면 gitignored 영역(예: `members/cy/local_notes/`)에 둘 것.

원본 `shared/eda.ipynb` 자체는 셀 출력에 raw id가 박혀있을 수 있어 `.gitignore`로 보호 중. 신규 노트북도 동일 패턴 (`*.ipynb`) 적용됨.
