# Spike (Feb 27-29) Mechanism EDA

**작성일**: 2026-05-27 · **데이터**: train.parquet (2019-11-01 ~ 2020-02-29, 8.35M events) · **스크립트**: [eda_spike_mechanism.py](./eda_spike_mechanism.py)

---

## 동기

[cart-to-purchase EDA](./eda_cart_to_purchase_result.md) 에서 spike 기간 (Feb 27-29) cart-경유 비율이 통계적으로 유의하게 낮음을 발견 (3.1% vs pre-spike 5.3%).

가설 후보 4가지 검증:
- **A. 판촉/단발성 이벤트** — cart 폭증 + 가격 하락
- **B. 데이터셋 cutoff / logging artifact**
- **C. 신규 user 유입 (cold-start)**
- **D. 시즌성 / category mix 변화**

---

## 핵심 발견 — Feb 27 비정상 패턴

### 일별 event count (last 20 days)

```
Date          cart   purchase  view     total
Feb 17-26   40-80      0-6     60-80k   60-80k   ← 정상
Feb 27          0       668     6,533    7,201   ← ⚠️
Feb 28         15       410    33,819   34,244
Feb 29         20       359    39,383   39,762
```

**Feb 27 의 비정상 패턴**:
- **cart = 0** — 평소 50-180/일에서 갑자기 0
- **view 10x 폭락** — 65k → 6.5k
- **purchase 100x+ 폭증** — 1-6/일에서 668

**자연스러운 browse-driven 행동 패턴**:
```
view ↑ → cart ↑ → purchase ↑ (점진적 funnel)
```

**Feb 27 패턴**:
```
view ↓ 10x  +  cart = 0  +  purchase ↑ 100x
```

이는 **평소의 user behavior mix 로는 설명 불가**.

---

## 가설별 평가

| 가설 | 평가 | 근거 |
|---|---|---|
| A. 판촉/단발성 이벤트 (browse-driven) | ❌ **기각** | 일반 판촉이면 view + cart 도 증가해야. **반대로 감소** |
| **A'. 직접 구매 flow 캠페인 (push/reorder/정기배송)** | ⚠️ **가능** | view 없이 purchase 가능. 다만 668건이 다 직접 flow 인 건 매우 많음 |
| **B. 데이터셋 cutoff / logging artifact** | ⚠️ **가능** | 마지막 3일 다른 sampling/filtering 또는 원본 boundary artifact |
| C. 신규 user 유입 | ❌ 약함 | 신규 user 비율 7-9% (두드러지지 않음) |
| D. 시즌성/category mix 변화 | ⚠️ 부분 | electronics 폭증 (xiaomi/sony/iqos) but category mix 유지 |

**A' (직접 구매 flow) 가 가능한 이유**:

| Flow | View 기록? | Cart 기록? |
|---|---|---|
| Push notification 의 "지금 사기" 직링크 | skip 가능 | X |
| 1-Click reorder ("Buy it again") | X | X |
| Subscribe & Save / 정기배송 자동 결제 | X | X |
| Affiliate / 외부 deep link | 부분 | X |

→ 이런 flow 가 dominant 한 시기면 view ↓ + cart=0 + purchase ↑ 가능.

**그래도 비정상인 이유** (A' 만으로 설명 어려운 부분):

1. **평소 user mix 변화 너무 큼** — 보통도 일부 user 는 1-click reorder 로 사는데, view 가 비례해서 줄지 않음. Feb 27 만 mix 가 통째로 바뀌는 건 부자연스러움
2. **Magnitude** — push 캠페인 1-2개로 view 10x + cart 100% 감소는 어려움
3. **부분 회복 패턴** — Feb 28-29 의 cart 15, 20 / view 33k, 39k 부분 회복은 system level 변화의 점진적 복원에 더 가까움

---

## 추가 검증 결과

### First-view → purchase 시간 (impulse buy 검증)

| 기간 | n | median (분) | p25 | p75 |
|---|---:|---:|---:|---:|
| Pre-spike | 639 | 1,610 (~27시간) | 17.7 | 18,543 |
| Spike | 1,437 | 8,594 (**~6일**) | 1,290 | 25,517 |

**Spike 의 구매가 오히려 더 긴 lag**. Impulse buy 가설 ❌ 기각. spike 며칠 전 view 했던 item 을 spike 때 구매.

### 일별 평균 purchase price (할인 검증)

| 기간 | n | avg | median |
|---|---:|---:|---:|
| Pre-spike (Feb 7-26) | 작음 | 60-150 | 30-100 |
| Feb 27 | 668 | 121.5 | 59.2 |
| Feb 28 | 410 | 128.8 | 62.8 |
| Feb 29 | 359 | 122.4 | 50.9 |

**할인 패턴 없음**. 가격대 정상.

### 신규 user 비율

| 기간 | new_user_rate |
|---|---:|
| Pre-spike (last 20 days 평균) | 0-50% (n 작아 변동 큼) |
| Feb 27 | 7.5% |
| Feb 28 | 8.1% |
| Feb 29 | 9.2% |

**두드러진 cold-start spike 없음**. C 가설 약함.

### Brand 분포 (purchase 기준)

```
Pre-spike top:  respect(68), sony(58), xiaomi(53), samsung(31)
Spike top:      xiaomi(171), sony(122), iqos(96), samsung(81), apple(45)
```

Electronics/모바일/담배 비중 ↑. Brand mix 변화 있으나 cart=0 까지 설명 안 됨.

### Category 분포

```
Pre-spike top:  apparel.shoes(258), shoes.slipons(70), shoes.sandals(60)
Spike top:      apparel.shoes(465), shoes.slipons(215), shoes.keds(138)
```

Apparel 중심은 유지. Category mix 자체는 normal.

---

## Feb 27 비정상 패턴의 가능 원인

**확정 불가** (주최사만 앎). 가능한 시나리오 (확률 비슷):

| 시나리오 | 가능성 | 설명 |
|---|---|---|
| 1. 대규모 push/email 캠페인 → 직접 구매 flow dominant | ⭐⭐⭐ | View skip 되는 deep-link 구매 |
| 2. 대회 주최사의 데이터 전처리 차이 | ⭐⭐⭐ | 마지막 3일이 다른 sampling / filtering 로 만들어짐 |
| 3. 원본 데이터셋의 boundary artifact | ⭐⭐⭐ | 공개 dataset (REES46 등) 의 boundary 흔한 artifact |
| 4. Train/test split 시점 차이 | ⭐⭐ | 평가셋 (Mar 1-7) 위해 마지막 3일 다른 처리 |
| 5. 정기배송 일제 갱신 | ⭐⭐ | 월말 갱신이면 가능, dataset spec 에 명시 없음 |
| 6. 서버 장애 + 부분 복구 | ⭐⭐ | view/cart logging 서버 죽고 purchase 만 살아남 |
| 7. 의도적 difficulty 추가 | ⭐⭐ | cold-start / 짧은 history 시나리오 합성 |

**공통점**: 어느 시나리오든 **"이 시기의 cart 행동 신호가 약하다"** 라는 결론은 동일.

**확정 답**: 알 수 없음. 다만 **평소 user behavior mix 와는 다른 패턴** 인 것은 확실.

---

## 우리 작업에 미치는 영향

### 1. 평가셋 (Mar 1-7) 도 비슷한 패턴일 가능성

- public LB 점수가 이 anomaly 와 관련 있을 수 있음
- cart-bypass dominant 한 평가셋이면, cart_boost 같은 휴리스틱 안 좋은 게 당연

### 2. 모델 평가의 일반성

- 우리 모델이 잘하는 것 = 이 anomaly 에서 잘 예측하는 것
- **다른 e-commerce 데이터 (정상 cart 패턴) 로 일반화 보장 X**
- Production 배포 시 다른 환경에서 행동 다를 수 있음

### 3. 모델 비교의 노이즈

- +0.0014 같은 작은 차이는 **이 anomaly 의 sensitivity** 일 수 있음
- 다른 데이터셋에선 sign 반대일 수도 (cart_boost: true 가 더 좋을 수도)

---

## `cart_boost: false` 의 최종 mechanism

```
1. 마지막 3일 cart 이벤트 거의 없음 (0, 15, 20)
2. 평소도 변환율 5% 수준 (전체 평균 3.8%)
3. cart_boost 가 boost 할 "carted-but-not-purchased" item 거의 없음 (특히 spike day)
4. ranking model (TIFU 64% weight) 이 이미 충분히 정밀
5. cart_boost 끄면 noise 작은 후처리 안 함 → marginal lift +0.0014
```

**+0.0014 의 의미**: 진짜 model improvement 가 아니라 **"데이터 anomaly + model 신뢰" 의 부작용 줄임**. Competition-specific quirk.

---

## Production / 산업 관점

### Lesson 1: 데이터 anomaly detection 필요

- 일별 event funnel ratio 자동 monitoring (view→cart→purchase 비율)
- 갑작스러운 break 감지 → alert
- Production 의 data drift detection 와 동일 원리

### Lesson 2: LB 점수 변화의 mechanism 검증

- 단순 hyperparameter 조정으로 +0.001 lift = **noise 일 가능성 높음**
- 진짜 model improvement vs data quirk 분리 = EDA 의 핵심 역할

### Lesson 3: Production sanity check

- Pinterest, Coupang 등 모두 daily anomaly detection 운영
- "Yesterday's cart count was 0" = 즉시 alert
- 이 데이터의 Feb 27 패턴이 실제 production 이면 즉시 incident

---

## Mature engineer talking point

> "팀원이 cart_boost flag 끄고 점수 +0.0014 얻음. 단순 hyperparameter sensitivity 인지 진짜 mechanism 인지 EDA 로 검증.
> 
> 검증 결과:
> 1. Feb 27 의 cart=0, view 10x 폭락, purchase 100x 폭증 — **평소 user behavior 와 다른 패턴**
> 2. 가능 원인: push 캠페인 / 데이터 전처리 / 원본 dataset artifact / 서버 부분 장애 — 확정 불가
> 3. 어느 경우든 **이 시기 cart 신호가 약함** → cart_boost 휴리스틱 작동 불가
> 4. +0.0014 는 noise removal 수준, **competition-specific quirk**
> 
> Production lesson: **점수 변화의 mechanism 확인 전에 'feature 가 좋다' 라고 결론짓지 말 것**. 데이터 quirk 일 수도 있음. data drift / sanity check 자동화가 production 의 표준."

---

## 후속 액션

| 옵션 | 가치 |
|---|---|
| 멘토 미팅에서 질문 ("이 데이터셋이 REES46 기반? 이런 artifact 산업에선 어떻게 detect?") | ⭐⭐⭐ |
| 추가 검증 (시간대별 anomaly start point) | ⭐ |
| 그냥 portfolio talking point 로 활용 | ⭐⭐⭐ |

**추천**: 멘토 질문 1개 + portfolio 활용. 깊이 파지 않음.