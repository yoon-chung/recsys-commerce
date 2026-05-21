# ensemble_v1_als_ease — RRF combine of ALS + EASE

Week 1 plan 의 첫 ensemble 체크포인트. 두 single-model 결과 (exp_000 ALS / exp_001 EASE) 를 **Reciprocal Rank Fusion** 으로 합쳐서, ensemble 파이프라인 검증 + 다양성 가설 (다른 family 모델 결합 시 lift) 확인.

## 가설

| 항목 | 근거 |
|---|---|
| RRF 가 두 모델 모두 대비 NDCG 향상 | exp_001 결과 — 동일 NDCG (0.1838 vs 0.1848) 인데 recall 은 EASE +13.7% → 두 모델이 잡는 정답 set 이 부분적으로 다름을 시사 |
| ALS (MF factor) ⊕ EASE (item-item) | 다른 family → 같은 item 에 대해 다른 rank 부여할 가능성 큼 |
| RRF k_const=60 (paper default) 충분 | top-50 후보만 사용하므로 k=60 이면 rank 1과 rank 50 의 기여 차이 약 50배. ablation 여지는 있음 |

## 입력 / 출력

**입력** (Week 1 single-model 결과 재활용):
- `../exp_000_als_baseline/predictions.parquet` — 638,257 × top-50
- `../exp_001_ease/predictions.parquet` — 623,866 × top-50 (cold-start 14,391 은 inference 단에서 popularity 로 채워졌으나, predictions.parquet 에는 알려진 user 만 들어있음)
- `../exp_001_ease/saved/val_gt.parquet`, `eval_users.json`, `mappings/` — 동일 split 재활용

**출력**:
- `fused_predictions.parquet` — RRF top-50 (`.gitignored`)
- `output.csv` — 제출용 (popularity fallback 적용, `.gitignored`)

## 알고리즘

```
for each user u:
    candidates = union of items from each model's top-50 for u
    for each candidate i:
        RRF(u, i) = sum over models m of  1 / (k_const + rank_m(u, i))
    sort candidates by RRF desc, keep top_n
```

- 한 모델만 추천한 item 도 정상 처리 (다른 모델 기여 0)
- rank 기반이라 모델 점수 scale calibration 불필요 (heterogeneous 모델에 안전)

## 실행 (서버에서)

```bash
cd /root/workspace/recsys-commerce && git pull

cd experiments/ensemble_v1_als_ease
python run.py
```

로컬에서 hyperparameter 만 바꿔 실험 시:

```bash
python run.py --k-const 30 --no-submission   # k_const ablation, 점수만 확인
python run.py --k-const 100 --no-submission
```

## 메트릭 (실행 후 작성)

| 모델 | NDCG@10 | recall@10 | NDCG 대비 |
|---|---:|---:|---:|
| exp_000 ALS | 0.1838 | 0.2558 | — |
| exp_001 EASE | 0.1848 | 0.2909 | +0.5% vs ALS |
| **RRF (k=60)** | TBD | TBD | — |

**판정 기준**:
- fused NDCG@10 > max(ALS, EASE) → RRF 파이프라인 성공, 5개 모델 확장 시 lift 기대
- fused NDCG@10 < min(ALS, EASE) → 두 모델 시그널이 정반대 (불일치) → 가설 재검토 (이론적으로 잘 안 일어남)
- fused 가 한쪽보다는 좋고 다른 쪽보다는 나쁨 → k_const ablation 가치

## 다음 액션

- 결과 OK → Week 1 Day 2-3 exp_002 BSARec 진입
- ALS+EASE 만으로 ALS 단독 lift 가 의미 있으면 **submission 제출 검토** (현재 동점 시 제출 횟수 규정 우려, calibration ratio 안정 시도)
- 후속 모델 추가될 때마다 `ensemble_vN_<list>/` 폴더로 누적 (`_v2`, `_v3` …)

## 참고

- Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*. SIGIR.
- [docs/candidate_models.md §5](../../docs/candidate_models.md) Week 1 5-모델 plan
- [exp_000_als_baseline/README.md](../exp_000_als_baseline/README.md), [exp_001_ease/README.md](../exp_001_ease/README.md)
