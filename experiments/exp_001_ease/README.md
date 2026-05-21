# exp_001_ease

**Embarrassingly Shallow Autoencoder (EASE, Steck 2019)** — closed-form item-item recommender.

Week 1 모델링 plan 의 Day 1 — 추천 시스템 입문 정수 + ALS 와 다른 family 로 ensemble 다양성 + Week 2 서비스용 fast inference foundation.

## 가설

1. **추천 시스템 입문 정수** — 30줄 closed-form. 알고리즘 자체가 단순해서 추천 시스템 입문 모델로 가치 큼
2. **EDA 정합** — item 반복도 14.6% ([eda_findings §7](../../docs/eda_findings.md)) + view-dominated implicit feedback (99.78%) 에서 item-item similarity 시그널 강함
3. **Ensemble 다양성** — ALS (MF factor 기반) vs EASE (item-item 기반). 두 family 결합 시 RRF lift 기대
4. **서비스 친화적** — 학습 후 inference 는 sparse × dense 행렬곱 1회 → 1 user 추론 < 1ms (Week 2 service 데모 적합)

## 알고리즘

```
1. Build sparse user × item matrix X (n_users, n_items)
2. G = X^T X                  # item × item Gram matrix
3. G += λI                    # ridge regularization (in-place)
4. P = inv(G + λI)            # Cholesky factor + solve (positive definite)
5. B = -P / diag(P)[None, :]  # column-wise divide by diagonal
6. fill_diagonal(B, 0)        # zero out self-loops
7. score(user, item) = (X[user] @ B)[item]
```

**시간 복잡도**: O(n_items²) memory for B + O(n_items³) for inversion = 29.5k³ ≈ 25T ops, 약 1-3분.

**메모리**: 29.5k² × 8B (float64 G/P) ≈ 7GB × 2 (peak during inversion) ≈ 14GB. Server 251GB RAM 여유. Local PC 16GB는 빠듯할 수 있음 → server 에서 학습 권장.

## 주요 hyperparameter

| 항목 | 값 | 비교 |
|---|---|---|
| `reg_lambda` | **200.0** | paper: MovieLens 100-1000 범위. 우리 데이터 sparse 정도 보고 ablation 가치 |
| `event_weights` | view 1 / cart 1 / purchase 1 | ALS exp_000 과 동일 (fair comparison) |
| `filter_already_liked` | **false** | exp_000 lesson — 본 거 다시 추천 가능 |
| `dtype_compute` | float64 | 행렬 inversion numerical stability |
| `dtype_storage` | float32 | B 저장 / 추론 속도 |
| `inference_batch_size` | 5000 | 5000 × 29.5k × 4B ≈ 590MB |

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull

cd experiments/exp_001_ease
python train.py
python inference.py
```

`core/` 모듈 자동 import (`Path(__file__).resolve().parents[2]` 트릭). wandb 는 `wandb_run_id.txt` 로 train+inference 단일 run 통합.

## 산출물

- `saved/B.npy` — item × item EASE 가중치 (float32, ~3.5GB) `.gitignored`
- `saved/interactions.npz` — training CSR
- `saved/mappings/` — user/item ID 매핑
- `saved/val_gt.parquet` — self-val ground truth
- `saved/eval_users.json` — eval target users
- `saved/wandb_run_id.txt` — train+inference run id 연결
- `predictions.parquet` — top-50 + score (ensemble 입력) `.gitignored`
- `output.csv` — 제출 파일 (638,257 × 10) `.gitignored`
- wandb artifact: B + predictions

## 결과 (2026-05-21 학습 완료)

| 메트릭 | 값 | exp_000 ALS | 차이 |
|---|---:|---:|---:|
| 자체 val NDCG@10 | **0.1848** | 0.1838 | +0.5% |
| 자체 val recall@10 | **0.2909** | 0.2558 | **+13.7%** |
| 공식 public NDCG@10 | (제출 안 함) | 0.0791 | — |

**소요 시간** (서버 RTX 3090, but EASE는 CPU only): 학습 ~4.7분 (Cholesky factor 38s + cho_solve 162s), 추론 8.6분 (628k user × 5k batch × 128 batch)

**B 행렬 통계** (λ=200): max=2.07, p99=4.9e-3, p50=6.4e-5 — λ 적절 (발산도 over-smoothing도 없음). ablation 가치 작음.

**해석**:
- NDCG 비등 + recall 큰 차이 → EASE는 정답을 top-10 안에 **더 자주** 잡지만 순위는 ALS와 비슷한 위치에 둠. item-item 시그널이 이 데이터에서 강함
- public 추정 (exp_000 calibration ratio 2.32): 0.1848 / 2.32 ≈ **0.0797**, ALS 0.0791과 사실상 동일

## ensemble_v1_als_ease 결과 (예상과 다름)

[ensemble_v1_als_ease/README.md](../ensemble_v1_als_ease/README.md) — RRF (k=60) 단순 fusion 결과:
- fused NDCG@10 = **0.1725** (vs ALS -0.011, vs EASE -0.012)
- fused recall@10 = 0.2624 (vs EASE -0.028)
- **두 모델 다 같은 CF family 라 다양성 부족** + EASE 가 ALS 를 거의 dominate → RRF 가 noise 추가. 다음 모델은 family-diverse (sequential / content) 가 더 가치 큼.

## 다음 액션

- ✅ ALS RRF 비교 완료 (negative result, family 다양성 학습)
- `reg_lambda` ablation — B stats 양호해서 우선순위 낮음
- **Week 1 Day 2-3 → exp_002 BSARec** (sequential, user-history 시간 시그널 → CF 와 진짜로 다른 family)

## 참고

- 모델 / 논문 / 코드 출처: [docs/references.md §1 — EASE row](../../docs/references.md#1-추천-모델--논문--공식-코드)
- [docs/candidate_models.md §5](../../docs/candidate_models.md) — Week 1 plan
- [docs/eda_findings.md §15](../../docs/eda_findings.md) — EDA 결과
