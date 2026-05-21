# service/

**Week 2 — production-style 추천 서비스 + LLM conversational layer**

[experiments/log.md 전략 pivot 섹션](../experiments/log.md#전략-pivot-2026-05-21--모델-lift--system--llm) 에 따라, Week 2 가 이 프로젝트의 **portfolio 무게중심**. Week 1 의 모델들을 통일 인터페이스로 wrap + 라이브 데모 가능한 service 로 패키징.

## 목표 (포트폴리오 talking points)

1. **Production system 운영 경험** — FastAPI 멀티 모델 서빙, latency SLO, fallback chain, observability
2. **LLM × Recsys 통합 (2025-26 트렌드)** — Solar API conversational layer
3. **현업 가치 작업** — 모델 lift 가 아닌 시스템 설계/운영 능력 시그널

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI service (app.py)                                            │
│                                                                       │
│  POST /predict                                                       │
│    body: { user_id, top_k, model?: 'auto'|'ease'|'bsarec'|'rrf' }   │
│    → predict_for_user() → [(item_id, score), ...]                    │
│                                                                       │
│  POST /chat                                                          │
│    body: { user_id, message }                                        │
│    → Solar (query understanding) → CF retrieval → Solar (응답 생성)  │
│    → { items: [...], reasoning: "..." }                              │
│                                                                       │
│  GET /healthz, /metrics (Prometheus)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Model Registry (model_registry.py)                                  │
│  - lazy-load exp_001/002/003 checkpoints on startup                  │
│  - 통일 predict_for_user(user_id, top_k) -> List[(item, score)]      │
│  - cold-start: popularity fallback                                   │
│  - error path: per-model timeout, fall back to next model in chain   │
├─────────────────────────────────────────────────────────────────────┤
│  Predictors                                                          │
│  ├─ EASEPredictor      (saved/B.npy + mappings)                     │
│  ├─ BSARecPredictor    (RecBole checkpoint)                         │
│  ├─ DiffRecPredictor   (RecBole checkpoint)                         │
│  ├─ RRFEnsemblePredictor (rank fusion of above)                     │
│  └─ PopularityPredictor (fallback)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  LLM Layer (llm_layer.py)                                            │
│  - Solar API client (Upstage)                                        │
│  - query_to_intent: 자연어 질의 → { category?, brand?, ... }         │
│  - generate_response: items + context → 자연어 추천 응답             │
└─────────────────────────────────────────────────────────────────────┘
```

## 파일 구조 (계획)

```
service/
├── app.py                # FastAPI 진입점 + 엔드포인트
├── model_registry.py     # 모델 로딩 + 라우팅 + fallback chain
├── predictors/           # 모델별 predict_for_user wrapper
│   ├── base.py            # Predictor 추상 클래스
│   ├── ease.py
│   ├── bsarec.py
│   ├── diffrec.py
│   ├── rrf_ensemble.py
│   └── popularity.py
├── llm_layer.py          # Solar API: intent 추출 + 응답 생성
├── observability.py      # Prometheus 메트릭 + structured log
├── config.py             # 모델 경로 / API 키 / SLO 등
├── requirements.txt
└── tests/
    ├── test_predict.py
    ├── test_chat.py
    └── test_fallback.py
```

## 통일 인터페이스

```python
# predictors/base.py
class Predictor(Protocol):
    def predict_for_user(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return [(item_id, score), ...] of length top_k.

        Cold-start (user_id 미학습) 시 빈 list 반환 → caller 가 fallback chain 적용.
        """
```

각 실험의 inference.py 핵심 로직을 추출해 Predictor 클래스로 wrap. saved/ 산출물 (예: exp_001/saved/B.npy, mappings/) 을 그대로 재활용.

## API 예시

**POST /predict** — 단순 추천
```json
{ "user_id": "abc-uuid", "top_k": 10, "model": "rrf" }
→
{
  "items": [
    {"item_id": "x-uuid", "score": 0.87},
    ...
  ],
  "model_used": "rrf",
  "latency_ms": 23
}
```

**POST /chat** — conversational 추천
```json
{ "user_id": "abc-uuid", "message": "겨울에 입을 따뜻한 외투 추천해줘" }
→
{
  "items": [...],
  "reasoning": "고객님이 최근 본 패션 카테고리 + 비슷한 취향의 사용자 패턴 기반...",
  "intent": { "category": "apparel.coat", "season": "winter" }
}
```

## Latency SLO (목표)

| Endpoint | P50 | P95 | P99 |
|---|---:|---:|---:|
| `/predict` (warm) | < 20ms | < 50ms | < 100ms |
| `/predict` (cold-start fallback) | < 50ms | < 100ms | < 200ms |
| `/chat` (Solar API 2회 호출) | < 1s | < 2.5s | < 5s |

## Fallback chain

```
predict request
   │
   ▼
[try requested model with timeout]
   │ fail / timeout
   ▼
[try next model in chain (e.g., EASE → RRF → BSARec)]
   │ all fail
   ▼
[popularity fallback (항상 응답)]
```

## Observability

- **Prometheus 메트릭**: 모델별 request count, latency histogram, error rate, cold-start hit rate
- **Structured log**: JSON 포맷, request_id 추적, model_used + latency_ms 매 응답
- **Health check**: `/healthz` 모델 로드 상태 검증

## 진행 순서 (Week 2)

| Day | 작업 |
|---|---|
| Week 2 Day 1 | `predictors/` base + EASE predictor + 단순 `/predict` 엔드포인트. 로컬에서 실 호출 데모 |
| Week 2 Day 2 | BSARec/DiffRec/RRF predictor 추가. 멀티 모델 라우팅 + cold-start fallback |
| Week 2 Day 3 | Latency 측정 + Prometheus 메트릭 + structured log 통합 |
| Week 2 Day 4 | **Solar API conversational layer** — `/chat` 엔드포인트. intent 추출 + 응답 생성 prompt 설계 |
| Week 2 Day 5 | 통합 테스트 + 데모 영상 녹화 (포트폴리오 자산) + README/docs 정리 |

## 핵심 의사결정 — 추후 결정 사항

- **서빙 형태**: 로컬 (uvicorn) 만으로 데모, OR 클라우드 배포? — 데모 영상이면 로컬 충분
- **모델 로드 시점**: startup (메모리 사용 ↑, latency ↓) vs lazy (반대) — startup 권장
- **EASE B 행렬 (3.5GB)** 메모리 상주 부담 — float16 quantization 또는 top-k 사전 계산 캐시 검토
- **Solar API rate limit / cost**: 데모용이면 무시, 본격 운영이면 캐시 layer 필요

## 의존

- [`core/`](../core/) — data_loader, validation, metrics, submission, ensemble (RRF)
- 각 [`experiments/exp_NNN_*/saved/`](../experiments/) — 학습된 모델 artifact (서버에서 로컬로 scp 또는 wandb artifact download)
- Solar API (Upstage) — 인증/quota 확인 필요

## 참고

- [experiments/log.md](../experiments/log.md) — Week 1 모델 결과 + 전략 pivot 배경
- [docs/references.md](../docs/references.md) — Solar API, FastAPI 등 외부 자료
