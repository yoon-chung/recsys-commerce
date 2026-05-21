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

## 결과 (학습 후 작성)

| 메트릭 | 값 | 비교 |
|---|---:|---|
| 자체 val NDCG@10 | TBD | exp_000 ALS 0.1838 |
| 자체 val recall@10 | TBD | exp_000 ALS 0.2558 |
| 공식 public NDCG@10 | (제출 안 함 — calibration 안정 후 결정) | — |

## 다음 액션

- 결과 측정 후 ALS 와 RRF 결합 시 lift 확인 (단순 평균 점수 비교)
- `reg_lambda` ablation 시간 여유 시 시도 (50 / 200 / 500 / 1000)
- Week 1 Day 2-3 → exp_002 BSARec (paper-to-code port)

## 참고

- Steck, H. (2019). "Embarrassingly Shallow Autoencoders for Sparse Data." WWW 2019.
- [docs/candidate_models.md §5](../../docs/candidate_models.md) Week 1 plan
- [docs/eda_findings.md §15](../../docs/eda_findings.md) EDA 결과
