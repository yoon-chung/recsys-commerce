# Cart → Purchase 변환율 EDA

**작성일**: 2026-05-27 · **데이터**: train.parquet (2019-11-01 ~ 2020-02-29, 8.35M events) · **스크립트**: [eda_cart_to_purchase.py](./eda_cart_to_purchase.py)

---

## 동기

팀원의 ensemble config 에서 `cart_boost: false` 설정으로 public LB 0.1490 → **0.1504** (+0.0014) 달성.

**검증 질문**: 왜 cart_boost 를 끄는 게 더 좋았나?

**가설** (1차): 마지막 3일 (Feb 27-29) 의 spike 기간 구매가 cart-bypass dominant 였기 때문.

---

## 방법

**Cart-bypass 정의**: purchase 시점 이전에 같은 (user, item) 의 cart 이벤트가 **하나도 없으면** "bypass purchase".

```python
purchases = df[event_type=="purchase"]
carts     = df[event_type=="cart"]
# 각 purchase 에 대해, 같은 (user, item) cart 가 이전에 있었는지
m = purchases.merge(carts, on=["user_id", "item_id"], how="left")
m["cart_before"] = m["cart_time"] < m["event_time"]
result = m.groupby(["user_id","item_id","event_time"])["cart_before"].any()
```

---

## 결과

### 1. 기간별 cart-bypass 비율

| 기간 | n_purchases | via cart | via_cart_rate | direct (bypass) | bypass_rate |
|---|---:|---:|---:|---:|---:|
| Pre-spike (Nov 1 ~ Feb 26) | 638 | 34 | 5.3% | 604 | **94.7%** |
| Spike (Feb 27-29) | 1,437 | 45 | 3.1% | 1,392 | **96.9%** |
| **전체** | **2,075** | **79** | **3.8%** | **1,996** | **96.2%** |

### 2. 일별 trend (last 20 days)

| 날짜 | n_purch | via_cart | via_cart_rate |
|---|---:|---:|---:|
| Feb 7-26 (20일) | 평균 ~3/일 | 0 | **0.00%** |
| 2020-02-27 (D-3) | 668 | 21 | 3.14% |
| 2020-02-28 (D-2) | 410 | 17 | 4.15% |
| 2020-02-29 (D-1) | 359 | 7 | 1.95% |

**Spike 분포**: 마지막 3일이 전체 purchase 의 **69%** (1,437 / 2,075) 차지.

---

## 핵심 발견 — 가설보다 큰 결론

### 1차 가설 검증

**가설**: spike 기간 (Feb 27-29) 이 cart-bypass dominant → **검증 O** (3.1% < 5.3%, -2.2%p)

### 더 큰 발견

**전체 데이터의 96% purchase 가 cart-bypass**. spike 든 아니든 거의 모든 구매가 cart 단계 없이 직접 발생.

- 일반 e-commerce 의 cart → purchase 변환율: 보통 **5-10%**
- 이 데이터: **3.8%** (매우 낮음)
- **Cart 의 buy-intent 신호가 약한 dataset**

---

## `cart_boost: false` 가 좋아진 진짜 이유

### Cart_boost 코드 동작

```python
def _cart_boost(preds, df, ...):
    """carted-but-not-purchased 아이템을 예측 상위로 이동."""
    cart_items = carted[uid] - purchased[uid]
    # 이 item 들을 ensemble prediction 의 맨 앞으로 강제 이동
```

→ 모델 score 계산엔 영향 X. **사후 ranking override** 만.

### 왜 끄는 게 좋아졌나

1. **"carted-but-not-purchased" 의 buy-intent 가 매우 약함**:
   - 이 데이터에서 cart 한 item 의 96% 는 결국 안 삼
   - 즉, 단순 호기심 cart / 비교용 cart 가 dominant
   - 이걸 무조건 top 으로 boost = **noise 를 추천 1순위로**

2. **모델 ensemble 이 이미 cart signal 활용 중**:
   - TIFU multi-behavior weight (cart event 가중)
   - MB-STR behavior embedding (cart 별도 임베딩)
   - BSARec/TiSASRec sequence (cart event 포함)
   - 사후 boost = **이중 가중 (redundant)**

3. **모델의 정밀 ranking 을 단순 휴리스틱이 override**:
   - TIFU 가중치 64% (압도적) → 이미 user repeat pattern 정확히 잡음
   - 단순 휴리스틱 "cart 한 item = 무조건 top" 이 이 정밀 ranking 손상

4. **Spike 기간에 더 두드러진 이유**:
   - spike 시기에 변환율 더 낮음 (3.1% vs 5.3%)
   - boost 의 misalignment 더 큼

---

## Production / 산업 관점

### 일반 패턴 vs 이 데이터

| 비교 | 일반 e-commerce | 이 데이터 |
|---|---|---|
| Cart → purchase 변환율 | 5-10% | 3.8% |
| Cart 의 buy-intent 신호 | 중간 | 약함 |
| Cart_boost 휴리스틱 효과 | 양수 (보통) | 음수 (이 경우) |

### Lesson

> **Model 이 충분히 강하면 휴리스틱이 오히려 noise**.

Pinterest TransAct, Alibaba DIN/BST 등 production system 모두 후처리 휴리스틱 줄이는 방향. 이번 case 동일 mechanism — 정밀 ranking model 의 결정을 신뢰하는 게 정답.

---

## Talking points

### Mature engineer signal

- **"끄니까 좋아졌다" → "왜 좋아졌나 데이터로 검증"**
- 가설 → EDA → 더 큰 발견 → mechanism 정확히 명명
- Competition 점수 변경이 noise 였는지 실제 인사이트였는지 데이터로 분리

### Service design 영향

우리 service 의 `/recommend` endpoint 에서도 같은 결정 필요:
- ❌ "cart 했지만 안 산 item 무조건 추천" 휴리스틱 추가 X
- ✅ LGBM reranker 의 ranking 신뢰
- 단, **"재구매 (#7)" 시나리오는 별개** — purchase history 가 cart history 보다 강한 신호

---

## 추가 검증 후보 (시간 남으면)

1. **Session 내 cart→purchase**: 같은 session 안에서 cart 직후 purchase 비율 (위 분석은 전 기간)
2. **Time-window cart→purchase**: cart 후 N분 / N시간 / N일 내 purchase
3. **Item 별 cart-conversion**: 특정 item 들은 변환율 높을 수도 (e.g., 식료품 vs 가전)
