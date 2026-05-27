# Stage 3 Service 구조 제안

**작성일**: 2026-05-27 · **상태**: 팀 논의용 초안

---

## 한 줄 요약

1 page 에 3 surface + LLM narration. **재구매 + LLM 톤** 이 우리 데이터 (반복 구매) 강점 살림.

---

## 핵심 3 surface + 보너스 1

| Surface | Use case | 출처 | LLM |
|---|---|---|---|
| **A** | 🎯 개인 맞춤 추천 | LGBM reranker (팀원 0.1504) | ⭐ 추천 이유 narration |
| **B** | 🔁 재구매 알림 | TIFU + 마지막 구매 시점 | ⭐ Timing 기반 톤 |
| **C** | 🛒 함께 자주 구매 | item-item co-purchase | ⭐ 매칭 reasoning |
| (보너스) D | 🏆 베스트셀러 | event_type==purchase 집계 | — |

---

## Single Page 구조

```
┌────────────────────────────────────────────────┐
│ 🛒 RecsysCommerce   (User U123)               │
├────────────────────────────────────────────────┤
│                                                │
│ ╔════════════════════════════════════════════╗ │
│ ║ 🔁 이제 다시 살 때입니다  (B: 재구매)      ║ │
│ ║ ┌──┬──┬──┐                                 ║ │
│ ║ │☕│🧴│🍫│                                 ║ │
│ ║ └──┴──┴──┘                                 ║ │
│ ║ 💬 "커피 마지막 구매 28일 전이에요"        ║ │
│ ╚════════════════════════════════════════════╝ │
│                                                │
│ ── 🎯 당신을 위한 추천 (A: 개인 맞춤) ────── │
│ ┌──┬──┬──┬──┬──┐                              │
│ │1 │2 │3 │4 │5 │  ← LGBM top-10              │
│ ├──┼──┼──┼──┼──┤                              │
│ │6 │7 │8 │9 │10│                              │
│ └──┴──┴──┴──┴──┘                              │
│ 💬 "겨울 outdoor 브랜드 선호 + 가격대 매칭"  │
│                                                │
│ ── 🏆 이번 주 베스트셀러 (D: 보너스) ────── │
│ ┌──┬──┬──┬──┬──┐                              │
│ │A │B │C │D │E │                              │
│ └──┴──┴──┴──┴──┘                              │
└────────────────────────────────────────────────┘

         ⬇ [아이템 클릭]

┌────────────────────────────────────────────────┐
│ ← 선택: Item #42 (Patagonia Jacket)           │
├────────────────────────────────────────────────┤
│ 🛒 함께 자주 구매되는 상품 (C: Co-purchase)   │
│ ┌──┬──┬──┬──┐                                  │
│ │🧣│🧤│👖│🧦│                                  │
│ └──┴──┴──┴──┘                                  │
│ 💬 "Patagonia 와 자주 함께 산 Uniqlo 가디건은  │
│     layering 용..."                            │
└────────────────────────────────────────────────┘
```

---

## Architecture

```
┌──────────┐    ┌─────────────────────────────────┐
│Streamlit │───►│ FastAPI Backend                  │
│  (UI)    │    │                                  │
└──────────┘    │ ┌──────────────────────────────┐ │
                │ │ Stage 1+2: LGBM rerank       │ │
                │ │   (팀원 0.1504 model)         │ │
                │ └──────────────────────────────┘ │
                │            ↓                     │
                │ ┌──────────────────────────────┐ │
                │ │ Stage 3: Solar Pro (LLM)     │ │
                │ │   streaming + intent routing │ │
                │ └──────────────────────────────┘ │
                └──────────────────────────────────┘
```

---

## API 4개

| Endpoint | 역할 |
|---|---|
| `POST /recommend` | A (개인 맞춤, LGBM top-10) |
| `POST /repurchase` | B (재구매 알림, TIFU + 마지막 구매 시점) |
| `GET /related/{item_id}` | C (item 클릭 시, co-purchase top-N) |
| `GET /bestseller` | D (보너스, 글로벌 popularity) |
| `POST /chat` | LLM streaming wrapper (위 결과를 grounding) |

---

## 산업 reference 매핑

| Reference | 우리 작업에 영향 |
|---|---|
| **AWS Personalize + Bedrock** | 추천 → LLM narration 표준 패턴 |
| **Amazon Rufus** | Intent routing, anti-hallucination grounding |
| **Coupang ML blog** | 한국 e-commerce production 패턴 (mid-size LLM, signal bus) |

---

## 차별점 (포트폴리오 talking point)

1. **재구매 (#7) + LLM 톤**: 우리 데이터 (반복 구매 중심) 강점 살림. 산업 5 use case 에 없는 추가 시나리오.
2. **2-stage backend + LLM frontend**: AWS Personalize + Bedrock 와 동일 구조. production-ready 패턴.
3. **Anti-hallucination grounding**: LLM 이 retrieved set 의 item 만 언급 → 검증 가능.

---

## 팀 논의 포인트

1. **재구매 banner 위치** — 맨 위? 별도 section?
2. **LLM 등장 빈도** — 3개 surface 모두? 핵심 1-2개만?
3. **UI 도구** — Streamlit (빠름) vs HTML+CSS (production-like)?
4. **A/B test 시뮬레이션** — 두 reranker (팀원 0.1490 vs cy 0.1353) 동시 노출?
5. **데모 시나리오** — 1-2명 대표 user 골라서 demo? 또는 user 선택 가능?

---

## 시간 예상

| 작업 | 시간 |
|---|---|
| FastAPI 4 endpoint scaffold | 4-6h |
| Co-purchase matrix builder | 1h |
| Solar Pro streaming + intent routing | 3-4h |
| Streamlit UI (single page) | 4-6h |
| Demo data prep + 영상 녹화 | 2h |
| **총** | **~15-20h** |