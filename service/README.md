# service/

**Week 2 — 추천 시스템 서비스 (예정)**

Week 1 ([experiments/](../experiments/)) 에서 학습한 모델들을 통일 인터페이스로 wrap 해 FastAPI 등으로 추천 API 제공.

## 계획 구조 (TBD)

```
service/
├── app.py              # FastAPI 진입점
├── model_registry.py   # experiments/exp_NNN/saved/ 로드 + wrap
├── predictor.py        # 통일 predict_for_user(user_id, top_k) 인터페이스
├── requirements.txt    # FastAPI / uvicorn 등
└── tests/              # API 동작 검증
```

## 통일 인터페이스 (Week 1 모델 wrap 시 합의)

```python
def predict_for_user(user_id: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Return [(item_id, score), ...] of length top_k.

    Cold-start (user_id 미학습) 시 popularity fallback.
    """
    ...
```

각 실험의 inference.py 를 service-friendly 모듈로 refactor 예정 (Week 1 종료 시점).

## 계획된 API 엔드포인트

```
GET  /predict/{user_id}?top_k=10
GET  /predict/{user_id}?top_k=10&model=ease     # 특정 모델 지정
GET  /predict/{user_id}?top_k=10&ensemble=rrf   # 모든 모델 RRF
POST /predict/batch                             # 여러 user 한 번에
```

## 의존 모듈

- [`core/`](../core/) — data_loader, validation, metrics, submission
- 각 [`experiments/exp_NNN/saved/`](../experiments/) — 학습된 모델 artifact

## 시작 시점

Week 1 모델링 완료 후 (~ 2026-05-28).
