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

## 핵심 발견 — Feb 27 데이터 anomaly

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

**자연스러운 user 행동 패턴**:
```
view ↑ → cart ↑ → purchase ↑ (점진적 funnel)
```

**Feb 27 패턴**:
```
view ↓ 10x  +  cart = 0  +  purchase ↑ 100x
```

view 와 cart 가 줄어들면서 purchase 가 폭증하는 건 **물리적으로 거의 불가능**. 모든 구매에는 어떤 형태로든 "보기" 가 선행되어야 함.

→ **데이터 logging 시스템 변경 / 수집 artifact** 강력한 신호.

---

## 가설별 평가

| 가설 | 평가 | 근거 |
|---|---|---|
| **A. 판촉/단발성 이벤트** | ❌ **기각** | 판촉이면 view + cart 도 증가해야. **반대로 감소** |
| **B. 데이터셋 cutoff / logging artifact** | ✅ **강력 지지** | Feb 27 cart=0 + view 10x 폭락 + purchase 100x 폭증 |
| C. 신규 user 유입 | ❌ 약함 | 신규 user 비율 7-9% (두드러지지 않음) |
| D. 시즌성/category mix 변화 | ⚠️ 부분 | electronics 폭증 (xiaomi/sony/iqos) but category mix 유지 |

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

## 데이터 anomaly 의 의미

### "데이터 logging 시스템 변경 / 수집 artifact" 의 정확한 뜻

**확정은 아닌 강력한 추정**. 가능한 시나리오 (확률 순):

| 시나리오 | 가능성 | 설명 |
|---|---|---|
| 1. 대회 주최사의 데이터 전처리 차이 | ⭐⭐⭐⭐ | 마지막 3일이 다른 sampling / filtering 로 만들어짐 |
| 2. 원본 데이터셋의 logging anomaly | ⭐⭐⭐⭐ | REES46 등 공개 dataset 의 boundary artifact (흔함) |
| 3. Train/test split 시점 차이 | ⭐⭐⭐ | 평가셋 (Mar 1-7) 위해 마지막 3일 다른 처리 |
| 4. 서버 장애 + 부분 복구 | ⭐⭐ | view/cart logging 서버 죽고 purchase 만 살아남 |
| 5. 의도적 difficulty 추가 | ⭐⭐ | cold-start / 짧은 history 시나리오 합성 |
| 6. 실제 단발성 promotion | ⭐ | cart=0 까지 설명 안 됨 |

**확정 답**: 모름 (주최사만 앎). **추정**: 데이터에 manipulation 이 있음 (의도적이든 artifact 든) 거의 확실.

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
> 1. Feb 27 의 cart=0, view 10x 폭락, purchase 100x 폭증 — **데이터 logging 시스템 변경의 artifact**
> 2. 자연스러운 판촉/spike 가 아니라 **데이터 수집 방식 변경의 결과**
> 3. Cart 이벤트가 없으니 cart_boost 휴리스틱 자체가 작동 안 함
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